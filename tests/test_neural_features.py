import unittest

import numpy as np

from src.neural.features import build_pair_features, log_mel


class NeuralFeatureTests(unittest.TestCase):
    def _build_features(self, query, candidate):
        return build_pair_features(
            query,
            candidate,
            sample_rate=11025,
            n_mels=32,
            n_fft=512,
            hop_length=256,
        )

    def test_build_pair_features_returns_three_matching_channels(self):
        query = np.sin(np.linspace(0, 1, 11025, dtype=np.float32))
        candidate = np.cos(np.linspace(0, 1, 11025, dtype=np.float32))

        features = self._build_features(query, candidate)

        self.assertEqual(3, features.shape[0])
        self.assertEqual(32, features.shape[1])
        self.assertEqual(features.shape[1:], features[2].shape)
        np.testing.assert_allclose(features[2], np.abs(features[0] - features[1]), rtol=1e-5)

    def test_silence_returns_finite_float32_features(self):
        audio = np.zeros(11025, dtype=np.float32)

        features = self._build_features(audio, audio)

        self.assertEqual(np.float32, features.dtype)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_log_mel_uses_scalar_per_window_normalization(self):
        audio = np.sin(np.linspace(0, 20, 11025, dtype=np.float32))

        features = log_mel(
            audio,
            sample_rate=11025,
            n_mels=32,
            n_fft=512,
            hop_length=256,
        )

        self.assertLess(abs(float(features.mean())), 1e-5)
        self.assertGreater(float(features.std()), 0.99)
        self.assertLess(float(features.std()), 1.01)
        self.assertFalse(np.allclose(features.mean(axis=0), 0.0, atol=1e-5))

    def test_mismatched_audio_lengths_are_cropped_to_common_frame_count(self):
        query = np.sin(np.linspace(0, 1, 11025, dtype=np.float32))
        candidate = np.cos(np.linspace(0, 1, 11281, dtype=np.float32))

        features = self._build_features(query, candidate)

        self.assertEqual((3, 32, 44), features.shape)
        np.testing.assert_allclose(features[2], np.abs(features[0] - features[1]), rtol=1e-5)

    def test_non_finite_input_raises_exception(self):
        query = np.zeros(11025, dtype=np.float32)
        candidate = np.zeros(11025, dtype=np.float32)
        query[0] = np.nan

        with self.assertRaises(Exception):
            self._build_features(query, candidate)


if __name__ == "__main__":
    unittest.main()
