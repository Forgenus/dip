import unittest

from src.recognition.search_trace import CandidateTrace, NeuralCandidateTrace, SearchTrace


class SearchTraceCandidateTests(unittest.TestCase):
    def test_search_trace_initializes_candidate_and_neural_lists(self):
        trace = SearchTrace()

        self.assertEqual([], trace.top_candidates)
        self.assertFalse(trace.neural_enabled)
        self.assertFalse(trace.neural_checked)
        self.assertEqual("", trace.neural_reason)
        self.assertEqual([], trace.neural_results)
        self.assertIsNone(trace.neural_error)

    def test_search_trace_preserves_existing_positional_constructor_order(self):
        trace = SearchTrace(-1, 2, 3)

        self.assertEqual(-1, trace.expected_id)
        self.assertEqual(2, trace.query_fp_count)
        self.assertEqual(3, trace.db_match_count)
        self.assertEqual([], trace.top_candidates)

    def test_candidate_trace_has_fingerprint_fields(self):
        candidate = CandidateTrace(
            song_id=7,
            rank=1,
            score=0.25,
            max_count=10,
            time_offset_bins=-3,
            time_offset_seconds=0.209,
        )

        self.assertEqual(7, candidate.song_id)
        self.assertEqual(1, candidate.rank)
        self.assertEqual(0.25, candidate.score)
        self.assertEqual(10, candidate.max_count)
        self.assertEqual(-3, candidate.time_offset_bins)
        self.assertEqual(0.209, candidate.time_offset_seconds)

    def test_neural_candidate_trace_has_shadow_fields(self):
        result = NeuralCandidateTrace(
            song_id=7,
            rank=1,
            fingerprint_score=0.25,
            fingerprint_max_count=10,
            fingerprint_time_offset_seconds=0.209,
            same_probability=0.82,
            decision="same",
            threshold=0.7,
            reliability="high",
            query_valid_seconds=5.0,
            candidate_valid_seconds=5.0,
            padding_ratio=0.0,
        )

        self.assertEqual("same", result.decision)
        self.assertEqual(0.82, result.same_probability)
        self.assertEqual("high", result.reliability)


if __name__ == "__main__":
    unittest.main()
