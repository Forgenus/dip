import unittest
from unittest.mock import patch

import numpy as np

import config as cfg
from src.recognition.query_pipeline import SearchResult
from src.recognition.search_trace import CandidateTrace, NeuralCandidateTrace, SearchTrace
from src.recognition.service import MusicRecognitionService


class FakePipeline:
    def search(self, audio, expected_id=-1, file_path=None):
        trace = SearchTrace()
        trace.top_candidates = [
            CandidateTrace(
                song_id=1,
                rank=1,
                score=0.2,
                max_count=5,
                time_offset_bins=0,
                time_offset_seconds=0.0,
            )
        ]
        trace.selected_id = 1
        trace.selected_score = 0.2
        trace.selected_max_count = 5
        return SearchResult(song_id=1, time_offset=0.0, trace=trace)


class FakeValidator:
    def evaluate_top_candidates(self, audio, candidates):
        return type(
            "Result",
            (),
            {
                "checked": True,
                "reason": "shadow_wide",
                "error": None,
                "results": [
                    NeuralCandidateTrace(
                        song_id=1,
                        rank=1,
                        fingerprint_score=0.2,
                        fingerprint_max_count=5,
                        fingerprint_time_offset_seconds=0.0,
                        same_probability=0.9,
                        decision="same",
                        threshold=0.7,
                        reliability="high",
                        query_valid_seconds=5.0,
                        candidate_valid_seconds=5.0,
                        padding_ratio=0.0,
                    )
                ],
            },
        )()


class RaisingValidator:
    def evaluate_top_candidates(self, audio, candidates):
        raise RuntimeError("neural unavailable")


class RecordingValidator:
    def __init__(self):
        self.candidates = None

    def evaluate_top_candidates(self, audio, candidates):
        self.candidates = list(candidates)
        return type(
            "Result",
            (),
            {
                "checked": True,
                "reason": "shadow_wide",
                "error": None,
                "results": [],
            },
        )()


class FakeOffsetPipeline:
    def __init__(self, results):
        self.results = list(results)

    def search(self, audio, expected_id=-1, file_path=None):
        return self.results.pop(0)


def search_result(song_id, score, time_offset, top_candidate_offset):
    trace = SearchTrace()
    trace.top_candidates = [
        CandidateTrace(
            song_id=song_id,
            rank=1,
            score=score,
            max_count=int(score * 1000),
            time_offset_bins=0,
            time_offset_seconds=top_candidate_offset,
        )
    ]
    trace.selected_id = song_id
    trace.selected_score = score
    trace.selected_max_count = int(score * 1000)
    return SearchResult(song_id=song_id, time_offset=time_offset, trace=trace)


class NeuralShadowServiceTests(unittest.TestCase):
    def make_service(self, validator):
        service = MusicRecognitionService.__new__(MusicRecognitionService)
        service.query_pipeline = FakePipeline()
        service.neural_validator = validator
        service.last_search_trace = None
        return service

    def test_shadow_validation_preserves_fingerprint_result(self):
        service = self.make_service(FakeValidator())

        song_id, time_offset = service.search_song(np.ones(30, dtype=np.float32), offset_fallback=False)

        self.assertEqual(1, song_id)
        self.assertEqual(0.0, time_offset)
        self.assertTrue(service.last_search_trace.neural_enabled)
        self.assertTrue(service.last_search_trace.neural_checked)
        self.assertEqual("shadow_wide", service.last_search_trace.neural_reason)
        self.assertEqual(1, len(service.last_search_trace.neural_results))

    def test_missing_validator_leaves_trace_unchecked(self):
        service = self.make_service(None)

        song_id, _ = service.search_song(np.ones(30, dtype=np.float32), offset_fallback=False)

        self.assertEqual(1, song_id)
        self.assertFalse(service.last_search_trace.neural_enabled)
        self.assertFalse(service.last_search_trace.neural_checked)

    def test_enabled_import_error_is_recorded_without_breaking_search(self):
        service = self.make_service(None)
        service.neural_validator_error = "torch import failed"

        with patch.object(cfg, "NEURAL_SHADOW_ENABLED", True, create=True):
            song_id, time_offset = service.search_song(
                np.ones(30, dtype=np.float32),
                offset_fallback=False,
            )

        self.assertEqual(1, song_id)
        self.assertEqual(0.0, time_offset)
        self.assertTrue(service.last_search_trace.neural_enabled)
        self.assertTrue(service.last_search_trace.neural_checked)
        self.assertEqual("shadow_wide", service.last_search_trace.neural_reason)
        self.assertEqual([], service.last_search_trace.neural_results)
        self.assertEqual("torch import failed", service.last_search_trace.neural_error)

    def test_runtime_enable_creates_enabled_validator(self):
        service = MusicRecognitionService.__new__(MusicRecognitionService)
        service.db = object()
        service.neural_validator = None
        service.neural_validator_error = None

        with patch("src.recognition.service.NeuralValidator") as validator_class:
            enabled = service.enable_neural_shadow()

        self.assertTrue(enabled)
        validator_class.assert_called_once_with(service.db, enabled=True)

    def test_validator_error_preserves_fingerprint_result_and_records_trace_error(self):
        service = self.make_service(RaisingValidator())

        song_id, time_offset = service.search_song(np.ones(30, dtype=np.float32), offset_fallback=False)

        self.assertEqual(1, song_id)
        self.assertEqual(0.0, time_offset)
        self.assertTrue(service.last_search_trace.neural_enabled)
        self.assertTrue(service.last_search_trace.neural_checked)
        self.assertEqual("shadow_wide", service.last_search_trace.neural_reason)
        self.assertEqual([], service.last_search_trace.neural_results)
        self.assertEqual("neural unavailable", service.last_search_trace.neural_error)

    def test_offset_fallback_passes_adjusted_top_candidate_offsets_to_validator(self):
        audio = np.arange(cfg.HOP_LENGTH * 4, dtype=np.float32)
        fallback_offset = cfg.HOP_LENGTH // 2
        fallback_offset_seconds = fallback_offset / cfg.SAMPLE_RATE
        pipeline = FakeOffsetPipeline(
            [
                search_result(song_id=1, score=0.01, time_offset=2.0, top_candidate_offset=2.0),
                search_result(song_id=2, score=0.08, time_offset=10.5, top_candidate_offset=10.5),
            ]
        )
        validator = RecordingValidator()
        service = self.make_service(validator)
        service.query_pipeline = pipeline

        with patch.object(cfg, "SEARCH_OFFSET_FALLBACK_MIN_SCORE", 0.03, create=True), patch.object(
            cfg,
            "SEARCH_OFFSET_FALLBACK_SAMPLES",
            [fallback_offset],
            create=True,
        ):
            song_id, time_offset = service.search_song(audio)

        self.assertEqual(2, song_id)
        self.assertAlmostEqual(10.5 - fallback_offset_seconds, time_offset)
        self.assertAlmostEqual(
            10.5 - fallback_offset_seconds,
            validator.candidates[0].time_offset_seconds,
        )


if __name__ == "__main__":
    unittest.main()
