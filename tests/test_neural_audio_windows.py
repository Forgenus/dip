import unittest

import numpy as np

from src.neural.audio_windows import crop_candidate_window, prepare_query_window


class NeuralAudioWindowTests(unittest.TestCase):
    def test_prepare_query_window_keeps_first_full_window(self):
        audio = np.arange(12, dtype=np.float32)

        window, meta = prepare_query_window(audio, sample_rate=2, window_seconds=5.0)

        np.testing.assert_array_equal(window, np.arange(10, dtype=np.float32))
        self.assertEqual(5.0, meta.valid_seconds)
        self.assertEqual(0.0, meta.padding_ratio)
        self.assertEqual("high", meta.reliability)
        self.assertFalse(meta.skipped)

    def test_prepare_query_window_zero_pads_medium_query(self):
        audio = np.arange(7, dtype=np.float32)

        window, meta = prepare_query_window(audio, sample_rate=2, window_seconds=5.0)

        np.testing.assert_array_equal(
            window,
            np.array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0], dtype=np.float32),
        )
        self.assertEqual(3.5, meta.valid_seconds)
        self.assertAlmostEqual(0.3, meta.padding_ratio)
        self.assertEqual("medium", meta.reliability)

    def test_prepare_query_window_skips_below_minimum(self):
        audio = np.arange(3, dtype=np.float32)

        window, meta = prepare_query_window(audio, sample_rate=2, window_seconds=5.0)

        self.assertEqual(10, len(window))
        self.assertEqual("skipped", meta.reliability)
        self.assertTrue(meta.skipped)

    def test_crop_candidate_window_pads_when_offset_is_near_start(self):
        audio = np.arange(6, dtype=np.float32)

        window, meta = crop_candidate_window(
            audio,
            sample_rate=2,
            start_seconds=-1.0,
            window_seconds=5.0,
        )

        np.testing.assert_array_equal(
            window,
            np.array([0, 0, 0, 1, 2, 3, 4, 5, 0, 0], dtype=np.float32),
        )
        self.assertEqual(3.0, meta.valid_seconds)
        self.assertAlmostEqual(0.4, meta.padding_ratio)

    def test_prepare_query_window_rejects_zero_sample_rate(self):
        audio = np.arange(6, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "sample_rate must be positive"):
            prepare_query_window(audio, sample_rate=0, window_seconds=5.0)

    def test_crop_candidate_window_rejects_zero_sample_rate(self):
        audio = np.arange(6, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "sample_rate must be positive"):
            crop_candidate_window(
                audio,
                sample_rate=0,
                start_seconds=0.0,
                window_seconds=5.0,
            )

    def test_prepare_query_window_returns_float32_for_integer_input(self):
        audio = np.arange(12, dtype=np.int16)

        window, _ = prepare_query_window(audio, sample_rate=2, window_seconds=5.0)

        self.assertEqual(np.float32, window.dtype)

    def test_crop_candidate_window_pads_past_end_with_zeros(self):
        audio = np.arange(6, dtype=np.float32)

        window, meta = crop_candidate_window(
            audio,
            sample_rate=2,
            start_seconds=2.0,
            window_seconds=3.0,
        )

        np.testing.assert_array_equal(
            window,
            np.array([4, 5, 0, 0, 0, 0], dtype=np.float32),
        )
        self.assertEqual(1.0, meta.valid_seconds)
        self.assertAlmostEqual(4 / 6, meta.padding_ratio)

    def test_crop_candidate_window_skips_fully_out_of_range_candidate(self):
        audio = np.arange(6, dtype=np.float32)

        window, meta = crop_candidate_window(
            audio,
            sample_rate=2,
            start_seconds=10.0,
            window_seconds=5.0,
        )

        np.testing.assert_array_equal(window, np.zeros(10, dtype=np.float32))
        self.assertEqual("skipped", meta.reliability)
        self.assertTrue(meta.skipped)

    def test_fractional_rounded_target_does_not_produce_negative_padding_ratio(self):
        audio = np.arange(2, dtype=np.float32)

        _, meta = prepare_query_window(audio, sample_rate=2, window_seconds=0.26)

        self.assertGreaterEqual(meta.padding_seconds, 0.0)
        self.assertGreaterEqual(meta.padding_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
