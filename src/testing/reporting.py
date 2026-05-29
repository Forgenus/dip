from dataclasses import dataclass
from pathlib import Path

from src.recognition.search_trace import NeuralCandidateTrace, SearchTrace
from src.testing.failure_analysis import FailureAnalysisReport


@dataclass
class FailedSnippetRecord:
    output_file: Path
    expected_id: int
    expected_title: str
    found_id: int
    found_title: str
    source_file: Path
    start_seconds: float
    duration_seconds: float


def format_search_trace(trace: SearchTrace) -> str:
    lines = [
        "=" * 72,
        "SEARCH TRACE",
        "-" * 72,
        f"dropped_stage={trace.dropped_stage}",
        f"reason={trace.reason}",
        "-" * 72,
        "lookup/filter",
        f"query_fp_count={trace.query_fp_count}",
        f"total_matches={trace.total_matches}",
        f"expected_raw_matches={trace.expected_raw_match_count}",
        f"expected_unique_addresses={trace.expected_unique_address_count}",
        "-" * 72,
        "selection/time coherency",
        f"selected_id={trace.selected_id}",
        f"selected_score={trace.selected_score:.4f}",
        f"expected_score={trace.expected_score if trace.expected_score is not None else 'None'}",
        f"selected_offset={trace.candidates_after_time.get(trace.selected_id)}",
        f"expected_offset={trace.candidates_after_time.get(trace.expected_id)}",
        f"expected_offset_buckets_top_10={dict(list(trace.expected_offset_buckets.items())[:10])}",
        f"candidates_after_filter={trace.candidates_after_filter[:10]}",
        f"candidates_after_time_first_10={dict(list(trace.candidates_after_time.items())[:10])}",
    ]
    neural_text = format_neural_shadow(trace)
    if neural_text:
        lines.extend(neural_text.splitlines())
    if trace.failure_analysis is not None:
        lines.extend(format_failure_analysis_report(trace.failure_analysis).splitlines())
    lines.append("=" * 72)
    return "\n".join(lines)


def format_neural_shadow(trace: SearchTrace) -> str:
    if not trace.neural_enabled and not trace.neural_checked and not trace.neural_results:
        return ""

    lines = [
        "-" * 72,
        "NEURAL SHADOW",
        "-" * 72,
        f"enabled={trace.neural_enabled}",
        f"checked={trace.neural_checked}",
        f"reason={trace.neural_reason}",
        f"error={trace.neural_error if trace.neural_error else 'None'}",
    ]
    if not trace.neural_results:
        lines.append("results=[]")
        return "\n".join(lines)

    for result in trace.neural_results:
        lines.append(format_neural_candidate(result))
    return "\n".join(lines)


def format_neural_candidate(result: NeuralCandidateTrace) -> str:
    return (
        f"rank={result.rank} song_id={result.song_id} "
        f"fp_score={result.fingerprint_score:.4f} "
        f"fp_max_count={result.fingerprint_max_count} "
        f"offset={result.fingerprint_time_offset_seconds:.4f}s "
        f"prob={result.same_probability:.6f} "
        f"decision={result.decision} "
        f"threshold={result.threshold:.2f} "
        f"reliability={result.reliability}"
    )


def neural_result_for_song(trace: SearchTrace, song_id: int) -> NeuralCandidateTrace | None:
    for result in trace.neural_results:
        if result.song_id == song_id:
            return result
    return None


def neural_validated_found_id(found_id: int, trace: SearchTrace | None) -> int:
    if trace is None or not trace.neural_checked or trace.neural_error or found_id == -1:
        return found_id

    result = neural_result_for_song(trace, found_id)
    if result is None:
        return found_id
    return found_id if result.decision == "same" else -1


def format_failure_analysis_report(report: FailureAnalysisReport) -> str:
    return "\n".join(
        [
            "-" * 72,
            "FAILED SNIPPET VS ORIGINAL SEGMENT",
            "-" * 72,
            f"analysis_error={report.error if report.error else 'None'}",
            "-" * 72,
            "peaks",
            f"snippet_points_count={report.snippet_points_count}",
            f"original_segment_points_count={report.original_segment_points_count}",
            f"common_points_count={report.common_points_count}",
            f"snippet_only_points_count={report.snippet_only_points_count}",
            f"original_only_points_count={report.original_only_points_count}",
            "-" * 72,
            "fingerprints",
            f"snippet_fp_count={report.snippet_fp_count}",
            f"original_segment_fp_count={report.original_segment_fp_count}",
            f"common_fingerprint_count={report.common_fingerprint_count}",
            f"common_address_count={report.common_address_count}",
            f"common_address_ratio={report.common_address_ratio:.4f}",
            "-" * 72,
            "coherent offsets",
            f"expected_offset_top_buckets={report.expected_offset_top_buckets}",
            "-" * 72,
            "SNIPPET VS FULL SONG INDEXING WINDOW",
            f"start_sample_mod_hop={report.start_sample_mod_hop}",
            f"full_song_start_frame={report.full_song_start_frame}",
            f"full_song_end_frame={report.full_song_end_frame}",
            f"full_song_window_fp_count={report.full_song_window_fp_count}",
            f"full_song_common_address_count={report.full_song_common_address_count}",
            f"full_song_common_address_ratio={report.full_song_common_address_ratio:.4f}",
            f"full_song_offset_top_buckets={report.full_song_offset_top_buckets}",
        ]
    )
