import unittest

import torch

from src.neural.model import PairClassifier


class PairClassifierTests(unittest.TestCase):
    def test_forward_returns_probability_per_pair(self):
        model = PairClassifier(input_channels=3)
        batch = torch.zeros((4, 3, 64, 80), dtype=torch.float32)

        output = model(batch)

        self.assertEqual((4,), tuple(output.shape))
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))

    def test_forward_preserves_batch_dimension_for_single_pair_in_eval_mode(self):
        model = PairClassifier(input_channels=3)
        model.eval()
        batch = torch.zeros((1, 3, 64, 80), dtype=torch.float32)

        output = model(batch)

        self.assertEqual((1,), tuple(output.shape))

    def test_rejects_non_positive_input_channels(self):
        with self.assertRaisesRegex(ValueError, "input_channels must be positive"):
            PairClassifier(input_channels=0)


if __name__ == "__main__":
    unittest.main()
