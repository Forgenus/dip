import unittest

import torch

from src.neural.features import TorchMelPairFeatureExtractor, build_symmetric_pair_features


class NeuralFeatureTests(unittest.TestCase):
    def test_build_symmetric_pair_features_returns_mean_and_abs_difference_channels(self):
        left = torch.ones((2, 4, 6), dtype=torch.float32)
        right = torch.full((2, 4, 6), 3.0, dtype=torch.float32)

        features = build_symmetric_pair_features(left, right)

        self.assertEqual((2, 2, 4, 6), tuple(features.shape))
        torch.testing.assert_close(torch.full((2, 4, 6), 2.0), features[:, 0])
        torch.testing.assert_close(torch.full((2, 4, 6), 2.0), features[:, 1])

    def test_build_symmetric_pair_features_is_swap_invariant(self):
        left = torch.randn((2, 8, 10), dtype=torch.float32)
        right = torch.randn((2, 8, 12), dtype=torch.float32)

        left_right = build_symmetric_pair_features(left, right)
        right_left = build_symmetric_pair_features(right, left)

        torch.testing.assert_close(left_right, right_left)

    def test_extractor_returns_finite_two_channel_features(self):
        extractor = TorchMelPairFeatureExtractor(
            sample_rate=11025,
            n_mels=16,
            n_fft=256,
            hop_length=128,
        )
        left = torch.sin(torch.linspace(0, 1, 11025, dtype=torch.float32))
        right = torch.cos(torch.linspace(0, 1, 11025, dtype=torch.float32))

        features = extractor(left, right)

        self.assertEqual(1, features.shape[0])
        self.assertEqual(2, features.shape[1])
        self.assertEqual(16, features.shape[2])
        self.assertTrue(torch.isfinite(features).all())

    def test_non_finite_input_raises_value_error(self):
        extractor = TorchMelPairFeatureExtractor(
            sample_rate=11025,
            n_mels=16,
            n_fft=256,
            hop_length=128,
        )
        left = torch.zeros(11025, dtype=torch.float32)
        right = torch.zeros(11025, dtype=torch.float32)
        left[0] = float("nan")

        with self.assertRaises(ValueError):
            extractor(left, right)


if __name__ == "__main__":
    unittest.main()
