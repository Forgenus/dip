import unittest
from unittest.mock import patch

import numpy as np

import config as cfg
from src.recognition.query_pipeline import SearchResult
from src.recognition.search_trace import SearchTrace
from src.recognition.service import MusicRecognitionService


def result(song_id: int, score: float, time_offset: float = 0.0) -> SearchResult:
    trace = SearchTrace()
    trace.selected_id = song_id
    trace.selected_score = score
    trace.selected_max_count = int(score * 1000)
    return SearchResult(song_id=song_id, time_offset=time_offset, trace=trace)


class FakeQueryPipeline:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def search(self, audio, expected_id=-1, file_path=None):
        self.calls.append(np.asarray(audio).copy())
        return self.results.pop(0)


class SearchOffsetFallbackTests(unittest.TestCase):
    def make_service(self, pipeline):
        service = MusicRecognitionService.__new__(MusicRecognitionService)
        service.query_pipeline = pipeline
        service.last_search_trace = None
        return service

    def test_uses_best_non_hop_aligned_fallback_when_primary_result_is_weak(self):
        audio = np.arange(cfg.HOP_LENGTH * 4, dtype=np.float32)
        fallback_offset = cfg.HOP_LENGTH // 2
        pipeline = FakeQueryPipeline(
            [
                result(song_id=1, score=0.01, time_offset=10.0),
                result(song_id=2, score=0.08, time_offset=10.5),
            ]
        )
        service = self.make_service(pipeline)

        with patch.object(cfg, "SEARCH_OFFSET_FALLBACK_MIN_SCORE", 0.03, create=True), patch.object(
            cfg,
            "SEARCH_OFFSET_FALLBACK_SAMPLES",
            [fallback_offset],
            create=True,
        ):
            song_id, time_offset = service.search_song(audio)

        self.assertEqual(2, song_id)
        self.assertEqual(2, len(pipeline.calls))
        np.testing.assert_array_equal(pipeline.calls[0], audio)
        np.testing.assert_array_equal(pipeline.calls[1], audio[fallback_offset:])
        self.assertAlmostEqual(10.5 - fallback_offset / cfg.SAMPLE_RATE, time_offset)
        self.assertTrue(service.last_search_trace.offset_fallback_selected)
        self.assertEqual(fallback_offset, service.last_search_trace.attempt_sample_offset)

    def test_skips_fallback_when_disabled(self):
        audio = np.arange(cfg.HOP_LENGTH * 4, dtype=np.float32)
        pipeline = FakeQueryPipeline([result(song_id=-1, score=0.0, time_offset=-1.0)])
        service = self.make_service(pipeline)

        song_id, time_offset = service.search_song(audio, offset_fallback=False)

        self.assertEqual(-1, song_id)
        self.assertEqual(-1.0, time_offset)
        self.assertEqual(1, len(pipeline.calls))

    def test_does_not_run_fallback_when_primary_result_is_strong(self):
        audio = np.arange(cfg.HOP_LENGTH * 4, dtype=np.float32)
        pipeline = FakeQueryPipeline([result(song_id=7, score=0.2, time_offset=3.0)])
        service = self.make_service(pipeline)

        with patch.object(cfg, "SEARCH_OFFSET_FALLBACK_MIN_SCORE", 0.03, create=True):
            song_id, time_offset = service.search_song(audio)

        self.assertEqual(7, song_id)
        self.assertEqual(3.0, time_offset)
        self.assertEqual(1, len(pipeline.calls))


if __name__ == "__main__":
    unittest.main()
