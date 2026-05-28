import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import config as cfg
from src.recognition.search_trace import SearchTrace
from src.testing.failure_analysis import empty_report
from src.testing.snippets import AudioSnippet
from src.testing.test_runner import TestRunner


class FakeService:
    def __init__(self):
        self.last_search_trace = SearchTrace()

    def get_random_song(self, rng):
        return {
            "song_id": 1,
            "title": "Expected",
            "file_path": Path("song.wav"),
        }

    def search_song(self, audio, _debug_correct_id=-1, file_path=None, offset_fallback=True):
        return 2, 0.0

    def get_song_by_id(self, song_id):
        return {"title": "Wrong"} if song_id == 2 else None


def runner_args(failure_analysis=False):
    return SimpleNamespace(
        reset_db=False,
        keep_rng=False,
        test_count=1,
        snippet_duration=8,
        failed_snippets_dir=Path("failed"),
        test_snippets_dir=Path("test"),
        save_test_snippets=False,
        align_snippet_start=False,
        noise=False,
        noise_level=0.02,
        volume="none",
        volume_factor=1.5,
        time_stretch_rate=1.0,
        offset_fallback=True,
        failure_analysis=failure_analysis,
    )


class TestRunnerFailureAnalysisTests(unittest.TestCase):
    def test_prints_hop_mod_in_basic_result_output(self):
        snippet = AudioSnippet(
            audio=np.array([0.1, -0.2], dtype=np.float32),
            source_file=Path("song.wav"),
            start_sample=cfg.HOP_LENGTH + 3,
            start_seconds=0.0,
            duration_seconds=1.0,
            sample_rate=11025,
        )

        with patch("src.testing.test_runner.create_snippet_from_file", return_value=snippet), patch(
            "src.testing.test_runner.save_snippet_to_file",
        ), patch("builtins.print") as print_mock:
            TestRunner(FakeService()).run(runner_args())

        output = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("hop_mod=3", output)

    def test_does_not_run_failure_analysis_by_default(self):
        snippet = AudioSnippet(
            audio=np.array([0.1, -0.2], dtype=np.float32),
            source_file=Path("song.wav"),
            start_sample=0,
            start_seconds=0.0,
            duration_seconds=1.0,
            sample_rate=11025,
        )

        with patch("src.testing.test_runner.create_snippet_from_file", return_value=snippet), patch(
            "src.testing.test_runner.save_snippet_to_file",
        ), patch("src.testing.test_runner.analyze_failed_snippet_against_file") as analyze:
            TestRunner(FakeService()).run(runner_args())

        analyze.assert_not_called()

    def test_runs_failure_analysis_when_flag_is_enabled(self):
        snippet = AudioSnippet(
            audio=np.array([0.1, -0.2], dtype=np.float32),
            source_file=Path("song.wav"),
            start_sample=0,
            start_seconds=0.0,
            duration_seconds=1.0,
            sample_rate=11025,
        )

        with patch("src.testing.test_runner.create_snippet_from_file", return_value=snippet), patch(
            "src.testing.test_runner.save_snippet_to_file",
        ), patch(
            "src.testing.test_runner.analyze_failed_snippet_against_file",
            return_value=empty_report("test"),
        ) as analyze:
            TestRunner(FakeService()).run(runner_args(failure_analysis=True))

        analyze.assert_called_once()


if __name__ == "__main__":
    unittest.main()
