import unittest

import torch

from src.neural.model import PairClassifier


class PairClassifierTests(unittest.TestCase):
    def test_forward_returns_logit_per_pair(self):
        model = PairClassifier(input_channels=2)
        batch = torch.zeros((4, 2, 80, 144), dtype=torch.float32)

        output = model(batch)

        self.assertEqual((4,), tuple(output.shape))
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_preserves_batch_dimension_for_single_pair_in_eval_mode(self):
        model = PairClassifier(input_channels=2)
        model.eval()
        batch = torch.zeros((1, 2, 80, 144), dtype=torch.float32)

        output = model(batch)

        self.assertEqual((1,), tuple(output.shape))

    def test_rejects_non_positive_input_channels(self):
        with self.assertRaisesRegex(ValueError, "input_channels must be a positive integer"):
            PairClassifier(input_channels=0)


if __name__ == "__main__":
    unittest.main()
