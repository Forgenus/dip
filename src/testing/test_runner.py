# test_runner.py
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

import config as cfg
from src.recognition.service import MusicRecognitionService
from src.testing.failure_analysis import analyze_failed_snippet_against_file
from src.testing.reporting import (
    FailedSnippetRecord,
    format_search_trace,
    neural_result_for_song,
    neural_validated_found_id,
)
from src.testing.snippet_effects import SnippetEffects, apply_snippet_effects
from src.testing.snippets import (
    build_failed_snippet_filename,
    build_test_snippet_filename,
    create_snippet_from_file,
    save_snippet_to_file,
)


class TestRunner:
    rng: np.random.Generator = np.random.default_rng()
    def __init__(self, service: MusicRecognitionService):
        self.service = service

    def run(self, args):
        if args.reset_db:
            self.service.clear_all()
            self.service.add_songs_from_folder(cfg.SONGS_DIR, max_amount=args.max_files)

        if args.keep_rng:
            TestRunner.rng = np.random.default_rng(cfg.RNG_SEED)

        correct = 0
        count = args.test_count
        total_time = 0.0
        offset_seconds_list: list[float] = []
        failed_records: list[FailedSnippetRecord] = []
        neural_stats = NeuralShadowStats()
        failed_snippets_dir = getattr(args, "failed_snippets_dir", cfg.FAILED_SNIPPETS_DIR)
        test_snippets_dir = getattr(args, "test_snippets_dir", cfg.TEST_SNIPPETS_DIR)
        snippet_effects = self._snippet_effects_from_args(args)

        for index in range(1, count + 1):
            song = self.service.get_random_song(rng=TestRunner.rng)
            if song is None:
                print("No songs in database")
                break

            path = song['file_path']

            snippet = create_snippet_from_file(
                file_path=path,
                snippet_duration=args.snippet_duration,
                rng=TestRunner.rng,
                align_to_hop=getattr(args, "align_snippet_start", False),
            )
            snippet = self._apply_snippet_effects(snippet, snippet_effects)

            if getattr(args, "save_test_snippets", False):
                output_file = self._save_test_snippet(
                    snippet=snippet,
                    output_dir=test_snippets_dir,
                    index=index,
                    expected_id=song["song_id"],
                    expected_title=song["title"],
                )
                print(f"Saved test snippet: {output_file}")

            start = time.perf_counter()
            found_id, time_offset = self.service.search_song(
                snippet.audio,
                _debug_correct_id=song['song_id'],
                file_path=path,
                offset_fallback=getattr(args, "offset_fallback", True),
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed

            found = self.service.get_song_by_id(found_id)
            expected = song['title']
            result = found['title'] if found else 'none'
            match = found_id == song['song_id']
            trace = self.service.last_search_trace
            neural_stats.record(
                expected_id=song["song_id"],
                found_id=found_id,
                baseline_match=match,
                trace=trace,
            )
            hop_mod = snippet.start_sample % cfg.HOP_LENGTH
            # Время смещения в секундах
            offset_seconds = hop_mod / cfg.SAMPLE_RATE
            offset_seconds_list.append(offset_seconds)

            if match:
                correct += 1
                print(
                    f"Y expected={expected} {song['song_id']} | found={result} {found_id} "
                    f"hop_mod={hop_mod} offset_seconds={offset_seconds:.4f} "
                    f"{self._format_success_metrics()}"
                )
            else:
                print(
                    f"{'Y' if match else 'N'} expected={expected} {song['song_id']} | "
                    f"found={result} {found_id} hop_mod={hop_mod} offset_seconds={offset_seconds:.4f} "
                )
                record = self._save_failed_snippet(
                    snippet=snippet,
                    output_dir=failed_snippets_dir,
                    index=len(failed_records) + 1,
                    expected_id=song["song_id"],
                    expected_title=expected,
                    found_id=found_id,
                    found_title=result,
                )
                failed_records.append(record)
                print(f"Saved failed snippet: {record.output_file}")

                if trace is not None and getattr(args, "failure_analysis", False):
                    trace.failure_analysis = analyze_failed_snippet_against_file(
                        snippet_audio=snippet.audio,
                        original_file=snippet.source_file,
                        start_seconds=snippet.start_seconds,
                        duration_seconds=snippet.duration_seconds,
                    )
                if trace is not None:
                    print(format_search_trace(trace))

        avg_time = total_time / count if count > 0 else 0.0
        accuracy = (correct / count * 100) if count > 0 else 0.0

        print(f"\nAccuracy: {correct}/{count} ({accuracy:.1f}%)")
        summary = neural_stats.format_summary(count)
        if summary:
            print(summary)
        print(f"Average query time: {avg_time:.4f} sec")
        
        if offset_seconds_list:
            min_offset = np.min(offset_seconds_list)
            max_offset = np.max(offset_seconds_list)
            mean_offset = np.mean(offset_seconds_list)
            median_offset = np.median(offset_seconds_list)
            print(f"Offset seconds - min={min_offset:.4f} max={max_offset:.4f} mean={mean_offset:.4f} median={median_offset:.4f}")

    def _snippet_effects_from_args(self, args) -> SnippetEffects:
        return SnippetEffects(
            noise=getattr(args, "noise", False),
            noise_level=getattr(args, "noise_level", 0.02),
            volume=getattr(args, "volume", "none"),
            volume_factor=getattr(args, "volume_factor", 1.5),
            time_stretch_rate=getattr(args, "time_stretch_rate", 1.0),
        )

    def _apply_snippet_effects(self, snippet, effects: SnippetEffects):
        audio = apply_snippet_effects(
            audio=snippet.audio,
            sample_rate=snippet.sample_rate,
            rng=TestRunner.rng,
            effects=effects,
        )
        return replace(
            snippet,
            audio=audio,
            duration_seconds=len(audio) / snippet.sample_rate,
        )

    def _format_success_metrics(self) -> str:
        trace = self.service.last_search_trace
        if trace is None:
            return "selected_score=None query_fp_count=None total_matches=None"

        selected_score = getattr(trace, "selected_score", None)
        if selected_score is None:
            score_text = "None"
        else:
            score_text = f"{selected_score:.4f}"

        return (
            f"selected_score={score_text} "
            f"query_fp_count={getattr(trace, 'query_fp_count', None)} "
            f"total_matches={getattr(trace, 'total_matches', None)}"
            f"max_count={getattr(trace, 'selected_max_count', None)}"
        )

    def _save_failed_snippet(
        self,
        snippet,
        output_dir,
        index: int,
        expected_id: int,
        expected_title: str,
        found_id: int,
        found_title: str,
    ) -> FailedSnippetRecord:
        filename = build_failed_snippet_filename(
            index=index,
            expected_id=expected_id,
            expected_title=expected_title,
            found_id=found_id,
            start_seconds=snippet.start_seconds,
        )
        output_file = Path(output_dir) / filename
        save_snippet_to_file(snippet, output_file)

        return FailedSnippetRecord(
            output_file=output_file,
            expected_id=expected_id,
            expected_title=expected_title,
            found_id=found_id,
            found_title=found_title,
            source_file=snippet.source_file,
            start_seconds=snippet.start_seconds,
            duration_seconds=snippet.duration_seconds,
        )

    def _save_test_snippet(
        self,
        snippet,
        output_dir,
        index: int,
        expected_id: int,
        expected_title: str,
    ) -> Path:
        filename = build_test_snippet_filename(
            index=index,
            expected_id=expected_id,
            expected_title=expected_title,
            start_seconds=snippet.start_seconds,
        )
        output_file = Path(output_dir) / filename
        save_snippet_to_file(snippet, output_file)
        return output_file


class NeuralShadowStats:
    def __init__(self) -> None:
        self.checked = 0
        self.errors = 0
        self.baseline_correct = 0
        self.simulated_correct = 0
        self.selected_same = 0
        self.selected_rejected = 0
        self.rejected_correct = 0
        self.rejected_incorrect = 0
        self.missing_selected_decision = 0

    def record(self, expected_id: int, found_id: int, baseline_match: bool, trace) -> None:
        if baseline_match:
            self.baseline_correct += 1

        if trace is None or not getattr(trace, "neural_checked", False):
            if baseline_match:
                self.simulated_correct += 1
            return

        self.checked += 1
        if getattr(trace, "neural_error", None):
            self.errors += 1
            if baseline_match:
                self.simulated_correct += 1
            return

        neural_found_id = neural_validated_found_id(found_id, trace)
        if neural_found_id == expected_id:
            self.simulated_correct += 1

        selected_result = neural_result_for_song(trace, found_id) if found_id != -1 else None
        if selected_result is None:
            self.missing_selected_decision += 1
            return

        if selected_result.decision == "same":
            self.selected_same += 1
        else:
            self.selected_rejected += 1
            if baseline_match:
                self.rejected_correct += 1
            else:
                self.rejected_incorrect += 1

    def format_summary(self, total_count: int) -> str:
        if self.checked == 0 and self.errors == 0:
            return ""

        baseline_accuracy = (
            self.baseline_correct / total_count * 100
            if total_count
            else 0.0
        )
        simulated_accuracy = (
            self.simulated_correct / total_count * 100
            if total_count
            else 0.0
        )
        delta = self.simulated_correct - self.baseline_correct

        return "\n".join(
            [
                "Neural shadow summary:",
                f"  checked={self.checked} errors={self.errors}",
                (
                    f"  baseline_accuracy={self.baseline_correct}/{total_count} "
                    f"({baseline_accuracy:.1f}%)"
                ),
                (
                    f"  simulated_validator_accuracy={self.simulated_correct}/{total_count} "
                    f"({simulated_accuracy:.1f}%) delta_correct={delta:+d}"
                ),
                (
                    f"  selected_same={self.selected_same} "
                    f"selected_rejected={self.selected_rejected} "
                    f"missing_selected_decision={self.missing_selected_decision}"
                ),
                (
                    f"  rejected_correct={self.rejected_correct} "
                    f"rejected_incorrect={self.rejected_incorrect}"
                ),
            ]
        )
