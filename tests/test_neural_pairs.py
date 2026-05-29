import unittest

import numpy as np

from src.neural.pairs import choose_query_valid_seconds, sample_pair_kind


class NeuralPairSamplingTests(unittest.TestCase):
    def test_choose_query_valid_seconds_is_deterministic_for_rng_seed(self):
        left_rng = np.random.default_rng(42)
        right_rng = np.random.default_rng(42)

        left = [choose_query_valid_seconds(left_rng) for _ in range(20)]
        right = [choose_query_valid_seconds(right_rng) for _ in range(20)]

        self.assertEqual(left, right)

    def test_choose_query_valid_seconds_prefers_full_duration(self):
        rng = np.random.default_rng(10)

        durations = [choose_query_valid_seconds(rng) for _ in range(1000)]

        full_count = sum(duration == 5.0 for duration in durations)
        medium_count = sum(3.0 <= duration < 5.0 for duration in durations)
        short_count = sum(2.0 <= duration < 3.0 for duration in durations)
        self.assertGreater(full_count, medium_count)
        self.assertGreater(medium_count, short_count)
        self.assertEqual(1000, full_count + medium_count + short_count)

    def test_sample_pair_kind_uses_requested_positive_and_hard_negative_mix(self):
        rng = np.random.default_rng(123)

        kinds = [sample_pair_kind(rng, positive_ratio=0.5, hard_negative_ratio=0.2) for _ in range(2000)]

        self.assertIn("positive_same_time", kinds)
        self.assertIn("positive_jittered", kinds)
        self.assertIn("negative_random", kinds)
        self.assertIn("negative_hard", kinds)
        positive_count = kinds.count("positive_same_time") + kinds.count("positive_jittered")
        hard_count = kinds.count("negative_hard")
        self.assertGreater(positive_count, hard_count)
        self.assertGreater(kinds.count("negative_random"), hard_count)

    def test_sample_pair_kind_rejects_invalid_ratios(self):
        rng = np.random.default_rng(1)

        with self.assertRaises(ValueError):
            sample_pair_kind(rng, positive_ratio=-0.1, hard_negative_ratio=0.2)
        with self.assertRaises(ValueError):
            sample_pair_kind(rng, positive_ratio=0.5, hard_negative_ratio=1.2)


if __name__ == "__main__":
    unittest.main()
