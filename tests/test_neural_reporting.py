import unittest

from src.recognition.search_trace import NeuralCandidateTrace, SearchTrace
from src.testing.reporting import (
    format_neural_shadow,
    format_search_trace,
    neural_validated_found_id,
)


def neural_result(song_id: int, decision: str, probability: float = 0.8) -> NeuralCandidateTrace:
    return NeuralCandidateTrace(
        song_id=song_id,
        rank=1,
        fingerprint_score=0.2,
        fingerprint_max_count=5,
        fingerprint_time_offset_seconds=1.25,
        same_probability=probability,
        decision=decision,
        threshold=0.7,
        reliability="high",
        query_valid_seconds=5.0,
        candidate_valid_seconds=5.0,
        padding_ratio=0.0,
    )


class NeuralReportingTests(unittest.TestCase):
    def test_format_neural_shadow_lists_candidate_decisions(self):
        trace = SearchTrace()
        trace.neural_enabled = True
        trace.neural_checked = True
        trace.neural_reason = "shadow_wide"
        trace.neural_results = [neural_result(7, "same", 0.91)]

        output = format_neural_shadow(trace)

        self.assertIn("NEURAL SHADOW", output)
        self.assertIn("song_id=7", output)
        self.assertIn("prob=0.910000", output)

    def test_format_search_trace_includes_neural_shadow(self):
        trace = SearchTrace()
        trace.neural_enabled = True
        trace.neural_checked = True
        trace.neural_results = [neural_result(7, "not_same", 0.12)]

        output = format_search_trace(trace)

        self.assertIn("NEURAL SHADOW", output)
        self.assertIn("decision=not_same", output)

    def test_neural_validated_found_id_rejects_selected_not_same(self):
        trace = SearchTrace()
        trace.neural_checked = True
        trace.neural_results = [neural_result(7, "not_same")]

        self.assertEqual(-1, neural_validated_found_id(7, trace))

    def test_neural_validated_found_id_keeps_selected_same(self):
        trace = SearchTrace()
        trace.neural_checked = True
        trace.neural_results = [neural_result(7, "same")]

        self.assertEqual(7, neural_validated_found_id(7, trace))


if __name__ == "__main__":
    unittest.main()
