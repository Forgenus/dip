import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.neural.splits import SongSplit, SongSplitItem, save_song_split
from src.neural.training import TrainingConfig, run_training, save_checkpoint


def synthetic_audio(path, target_sr):
    duration = 8
    if str(path).endswith("two.wav") or str(path).endswith("heldout-two.wav"):
        return np.linspace(1.0, -1.0, target_sr * duration, dtype=np.float32)
    return np.linspace(-1.0, 1.0, target_sr * duration, dtype=np.float32)


class TinyPairClassifier(torch.nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(input_channels, 1)

    def forward(self, batch):
        pooled = self.pool(batch).flatten(1)
        return self.linear(pooled).squeeze(1)


class NeuralTrainingRunTests(unittest.TestCase):
    def make_split(self, path: Path) -> None:
        split = SongSplit(
            version=1,
            seed=123,
            ratios={"train": 0.5, "validation_heldout": 0.5, "test_heldout": 0.0},
            counts={"total": 4, "train": 2, "validation_heldout": 2, "test_heldout": 0},
            train=[
                SongSplitItem(1, "data/audio/one.wav", "One", "Artist", 8.0),
                SongSplitItem(2, "data/audio/two.wav", "Two", "Artist", 8.0),
            ],
            validation_heldout=[
                SongSplitItem(3, "data/audio/heldout-one.wav", "Three", "Artist", 8.0),
                SongSplitItem(4, "data/audio/heldout-two.wav", "Four", "Artist", 8.0),
            ],
            test_heldout=[],
        )
        save_song_split(split, path)

    def make_args(self, split_path: Path, device: str = "cpu") -> Namespace:
        return Namespace(
            split=split_path,
            epochs=1,
            batch_size=2,
            device=device,
            examples_per_epoch=2,
            validation_examples=2,
            num_workers=0,
            fresh=False,
        )

    @mock.patch("src.neural.training.PairClassifier", TinyPairClassifier, create=True)
    @mock.patch("src.neural.training_data.pp.load_audio", side_effect=synthetic_audio)
    def test_tiny_split_runs_one_epoch_and_writes_checkpoint(self, _load_audio):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            split_path = temp_path / "split.json"
            checkpoint_path = temp_path / "checkpoint.pt"
            self.make_split(split_path)

            with (
                mock.patch("src.neural.training.cfg.NEURAL_MODEL_PATH", checkpoint_path),
                mock.patch("src.neural.training.cfg.NEURAL_TRAIN_NUM_WORKERS", 0),
            ):
                result = run_training(self.make_args(split_path))

            self.assertEqual(0, result)
            self.assertTrue(checkpoint_path.exists())
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(1, checkpoint["epoch"])
            self.assertIn("model_state", checkpoint)

    @mock.patch("src.neural.training.PairClassifier", TinyPairClassifier, create=True)
    @mock.patch("src.neural.training_data.pp.load_audio", side_effect=synthetic_audio)
    def test_existing_checkpoint_is_resumed_by_default(self, _load_audio):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            split_path = temp_path / "split.json"
            checkpoint_path = temp_path / "checkpoint.pt"
            self.make_split(split_path)

            model = TinyPairClassifier(input_channels=2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=5,
                config=TrainingConfig(epochs=1, batch_size=2),
                best_metric=999.0,
            )

            with mock.patch("src.neural.training.cfg.NEURAL_MODEL_PATH", checkpoint_path):
                result = run_training(self.make_args(split_path))

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(0, result)
            self.assertEqual(6, checkpoint["epoch"])

    @mock.patch("src.neural.training.load_checkpoint")
    @mock.patch("src.neural.training.PairClassifier", TinyPairClassifier, create=True)
    @mock.patch("src.neural.training_data.pp.load_audio", side_effect=synthetic_audio)
    def test_fresh_skips_existing_checkpoint(self, _load_audio, load_checkpoint):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            split_path = temp_path / "split.json"
            checkpoint_path = temp_path / "checkpoint.pt"
            checkpoint_path.write_bytes(b"existing checkpoint")
            self.make_split(split_path)

            args = self.make_args(split_path)
            args.fresh = True

            with mock.patch("src.neural.training.cfg.NEURAL_MODEL_PATH", checkpoint_path):
                result = run_training(args)

            self.assertEqual(0, result)
            load_checkpoint.assert_not_called()

    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_cuda_requested_but_unavailable_returns_1(self, _is_available):
        with tempfile.TemporaryDirectory() as temp_dir:
            split_path = Path(temp_dir) / "split.json"
            self.make_split(split_path)

            result = run_training(self.make_args(split_path, device="cuda"))

        self.assertEqual(1, result)

    @mock.patch("src.neural.training.save_checkpoint")
    @mock.patch("src.neural.training.evaluate_loader", return_value={"loss": 1.0})
    @mock.patch("src.neural.training.train_one_epoch", return_value=1.0)
    @mock.patch("src.neural.training.make_training_loader", side_effect=lambda dataset, **_kwargs: dataset)
    @mock.patch("src.neural.training.NeuralPairDataset")
    def test_validation_datasets_disable_query_augmentation(
        self,
        dataset_class,
        _make_loader,
        _train_one_epoch,
        _evaluate_loader,
        _save_checkpoint,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            split_path = Path(temp_dir) / "split.json"
            self.make_split(split_path)

            result = run_training(self.make_args(split_path))

        self.assertEqual(0, result)
        self.assertEqual(3, dataset_class.call_count)
        train_config = dataset_class.call_args_list[0].kwargs.get("query_augmentation_config")
        known_config = dataset_class.call_args_list[1].kwargs.get("query_augmentation_config")
        heldout_config = dataset_class.call_args_list[2].kwargs.get("query_augmentation_config")
        self.assertIsNone(train_config)
        self.assertFalse(known_config.enabled)
        self.assertFalse(heldout_config.enabled)


if __name__ == "__main__":
    unittest.main()
