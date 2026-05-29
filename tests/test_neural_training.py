import tempfile
import unittest
from pathlib import Path

import torch

import config as cfg
from src.neural.training import (
    TrainingConfig,
    evaluate_logits,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


class NeuralTrainingTests(unittest.TestCase):
    def test_training_config_uses_project_defaults(self):
        config = TrainingConfig()

        self.assertEqual(cfg.NEURAL_TRAIN_BATCH_SIZE, config.batch_size)
        self.assertEqual(cfg.NEURAL_TRAIN_EPOCHS, config.epochs)
        self.assertEqual(cfg.NEURAL_SPLIT_PATH, config.split_path)

    def test_evaluate_logits_reports_probabilities_and_metadata(self):
        rows = evaluate_logits(
            torch.tensor([0.0, 2.0]),
            torch.tensor([0.0, 1.0]),
            ["negative_random", "positive_same_time"],
            ["5.0s", "3-4s"],
        )

        self.assertEqual(2, len(rows))
        self.assertAlmostEqual(0.5, rows[0]["probability"])
        self.assertEqual(0, rows[0]["label"])
        self.assertEqual("negative_random", rows[0]["pair_type"])
        self.assertEqual("3-4s", rows[1]["duration_bucket"])

    def test_train_one_epoch_uses_logits_loss_and_updates_model(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.weight.detach().clone()
        batches = [
            (
                torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
                torch.tensor([0.0, 1.0], dtype=torch.float32),
            )
        ]

        loss = train_one_epoch(model, batches, optimizer, torch.device("cpu"), mixed_precision=True)

        self.assertGreater(loss, 0.0)
        self.assertFalse(torch.equal(before, model.weight.detach()))

    def test_checkpoint_round_trips_model_optimizer_epoch_and_metric(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        config = TrainingConfig(batch_size=16, epochs=3)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, epoch=2, config=config, best_metric=0.42)

            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
            epoch, best_metric = load_checkpoint(path, restored_model, restored_optimizer)

        self.assertEqual(2, epoch)
        self.assertEqual(0.42, best_metric)
        for left, right in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
