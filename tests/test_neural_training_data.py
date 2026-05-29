import unittest
from unittest import mock

import numpy as np
import torch

from src.neural.augmentations import QueryAugmentationConfig, apply_query_augmentations
from src.neural.training_data import NeuralPairDataset, make_training_loader
from src.neural.splits import SongSplitItem


SONGS = [
    SongSplitItem(1, "data/audio/one.wav", "One", "Artist", 12.0),
    SongSplitItem(2, "data/audio/two.wav", "Two", "Artist", 14.0),
]


def synthetic_audio(path, target_sr):
    if str(path).endswith("one.wav"):
        return np.linspace(0.0, 1.0, target_sr * 12, dtype=np.float32)
    return np.linspace(1.0, 0.0, target_sr * 14, dtype=np.float32)


class NeuralTrainingDataTests(unittest.TestCase):
    def make_dataset(self, **overrides):
        params = {
            "items": SONGS,
            "examples_per_epoch": 4,
            "sample_rate": 8000,
            "window_seconds": 1.0,
            "n_mels": 8,
            "n_fft": 128,
            "hop_length": 64,
            "positive_ratio": 0.5,
            "hard_negative_ratio": 0.0,
            "positive_jitter_seconds": 0.1,
            "seed": 123,
        }
        params.update(overrides)
        return NeuralPairDataset(**params)

    @mock.patch("src.neural.training_data.pp.load_audio", side_effect=synthetic_audio)
    def test_item_has_feature_shape_and_metadata(self, _load_audio):
        dataset = self.make_dataset(examples_per_epoch=1, positive_ratio=1.0)

        item = dataset[0]

        self.assertEqual(1, len(dataset))
        self.assertEqual(2, item.features.shape[0])
        self.assertEqual(8, item.features.shape[1])
        self.assertGreater(item.features.shape[2], 0)
        self.assertEqual((), tuple(item.label.shape))
        self.assertIn(item.pair_type, {"positive_same_time", "positive_jittered"})
        self.assertIn(item.duration_bucket, {"5.0s", "4-5s", "3-4s", "2-3s"})

    def test_random_negative_uses_different_song_ids(self):
        dataset = self.make_dataset(positive_ratio=0.0, hard_negative_ratio=0.0)

        pair = dataset._sample_pair(0)

        self.assertEqual("negative_random", pair.pair_type)
        self.assertNotEqual(pair.query_song_id, pair.candidate_song_id)

    @mock.patch("src.neural.training_data.pp.load_audio", side_effect=synthetic_audio)
    def test_loader_collates_two_examples(self, _load_audio):
        dataset = self.make_dataset(examples_per_epoch=2, positive_ratio=1.0)
        loader = make_training_loader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader))

        self.assertEqual((2, 2, 8), tuple(batch["features"].shape[:3]))
        self.assertEqual((2,), tuple(batch["labels"].shape))
        self.assertEqual(2, len(batch["pair_type"]))
        self.assertEqual(2, len(batch["duration_bucket"]))

    @mock.patch("src.neural.training_data.apply_query_augmentations")
    @mock.patch("src.neural.training_data.pp.load_audio", side_effect=synthetic_audio)
    def test_dataset_applies_augmentations_to_query_before_features(
        self,
        _load_audio,
        apply_augmentations,
    ):
        apply_augmentations.side_effect = lambda audio, sample_rate, rng, config: audio + 0.1
        dataset = self.make_dataset(examples_per_epoch=1, positive_ratio=1.0)
        dataset.feature_extractor = mock.Mock(
            return_value=torch.zeros((1, 2, 8, 4), dtype=torch.float32)
        )

        dataset[0]

        apply_augmentations.assert_called_once()
        query_tensor = dataset.feature_extractor.call_args.args[0]
        self.assertTrue(torch.all(query_tensor >= 0.1))

    def test_query_augmentations_apply_noise_and_strong_volume(self):
        audio = np.ones(8000, dtype=np.float32) * 0.25
        rng = np.random.default_rng(4)
        config = QueryAugmentationConfig(
            noise_min=0.01,
            noise_max=0.03,
            volume_min=0.25,
            volume_max=3.0,
            time_stretch_min=1.0,
            time_stretch_max=1.0,
        )

        augmented = apply_query_augmentations(audio, sample_rate=8000, rng=rng, config=config)

        self.assertEqual(audio.shape, augmented.shape)
        self.assertFalse(np.allclose(audio, augmented))
        self.assertLessEqual(float(np.max(np.abs(augmented))), 1.0)

    def test_query_augmentations_can_be_disabled(self):
        audio = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)
        rng = np.random.default_rng(4)

        augmented = apply_query_augmentations(
            audio,
            sample_rate=8000,
            rng=rng,
            config=QueryAugmentationConfig(enabled=False),
        )

        np.testing.assert_array_equal(audio, augmented)


if __name__ == "__main__":
    unittest.main()
