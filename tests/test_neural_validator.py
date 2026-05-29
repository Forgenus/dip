import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from src.neural.model import PairClassifier
from src.neural.validator import NeuralValidator
from src.recognition.search_trace import CandidateTrace


class FakeModel:
    def eval(self):
        return self

    def __call__(self, batch):
        import torch
        return torch.tensor([2.0], dtype=torch.float32)


class LowProbabilityModel:
    def eval(self):
        return self

    def __call__(self, batch):
        import torch
        return torch.tensor([-2.0], dtype=torch.float32)


class FakeDb:
    def get_song_by_id(self, song_id):
        return {"song_id": song_id, "file_path": Path("candidate.wav")}


class EvalRaisesModel:
    def eval(self):
        raise AssertionError("model should not be evaluated")

    def __call__(self, batch):
        raise AssertionError("model should not be called")


class MissingSongDb:
    def get_song_by_id(self, song_id):
        return None


class NeuralValidatorTests(unittest.TestCase):
    def make_candidate(self):
        return CandidateTrace(
            song_id=1,
            rank=1,
            score=0.12,
            max_count=6,
            time_offset_bins=0,
            time_offset_seconds=0.0,
        )

    def test_evaluate_top_candidates_returns_trace_results(self):
        validator = NeuralValidator(
            db=FakeDb(),
            model=FakeModel(),
            enabled=True,
            threshold=0.7,
            top_n=1,
            sample_rate=4,
            window_seconds=5.0,
            n_mels=8,
            n_fft=8,
            hop_length=4,
        )
        query = np.ones(20, dtype=np.float32)
        candidate = self.make_candidate()

        with patch("src.neural.validator.pp.load_audio", return_value=np.ones(20, dtype=np.float32)):
            result = validator.evaluate_top_candidates(query, [candidate])

        self.assertTrue(result.checked)
        self.assertIsNone(result.error)
        self.assertEqual(1, len(result.results))
        self.assertEqual("same", result.results[0].decision)
        self.assertAlmostEqual(0.880797, result.results[0].same_probability, places=6)

    def test_below_threshold_decision_is_not_same(self):
        validator = NeuralValidator(
            db=FakeDb(),
            model=LowProbabilityModel(),
            enabled=True,
            threshold=0.7,
            top_n=1,
            sample_rate=4,
            window_seconds=5.0,
            n_mels=8,
            n_fft=8,
            hop_length=4,
        )

        with patch("src.neural.validator.pp.load_audio", return_value=np.ones(20, dtype=np.float32)):
            result = validator.evaluate_top_candidates(
                np.ones(20, dtype=np.float32),
                [self.make_candidate()],
            )

        self.assertEqual("not_same", result.results[0].decision)

    def test_disabled_validator_returns_unchecked_result(self):
        validator = NeuralValidator(db=FakeDb(), model=FakeModel(), enabled=False)

        result = validator.evaluate_top_candidates(np.ones(20, dtype=np.float32), [])

        self.assertFalse(result.checked)
        self.assertEqual([], result.results)
        self.assertIsNone(result.error)

    def test_load_model_accepts_training_checkpoint_wrapper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            model = PairClassifier(input_channels=2)
            torch.save({"model_state": model.state_dict(), "epoch": 3}, checkpoint_path)
            validator = NeuralValidator(db=FakeDb(), model_path=checkpoint_path)

            loaded = validator._load_model()

        self.assertIsInstance(loaded, PairClassifier)

    def test_load_model_accepts_checkpoint_with_path_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            model = PairClassifier(input_channels=2)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": 3,
                    "config": {"split_path": Path("data/neural/splits/song_split.json")},
                },
                checkpoint_path,
            )
            validator = NeuralValidator(db=FakeDb(), model_path=checkpoint_path)

            loaded = validator._load_model()

        self.assertIsInstance(loaded, PairClassifier)

    def test_enabled_no_candidates_returns_unchecked_without_evaluating_model(self):
        validator = NeuralValidator(db=FakeDb(), model=EvalRaisesModel(), enabled=True)

        result = validator.evaluate_top_candidates(np.ones(20, dtype=np.float32), [])

        self.assertFalse(result.checked)
        self.assertEqual("no_candidates", result.reason)
        self.assertEqual([], result.results)
        self.assertIsNone(result.error)

    def test_top_n_zero_returns_checked_empty_without_evaluating_model(self):
        validator = NeuralValidator(
            db=FakeDb(),
            model=EvalRaisesModel(),
            enabled=True,
            top_n=0,
            sample_rate=4,
            window_seconds=5.0,
        )

        result = validator.evaluate_top_candidates(
            np.ones(20, dtype=np.float32),
            [self.make_candidate()],
        )

        self.assertTrue(result.checked)
        self.assertEqual("shadow_wide", result.reason)
        self.assertEqual([], result.results)
        self.assertIsNone(result.error)

    def test_missing_song_returns_checked_empty_without_evaluating_model(self):
        validator = NeuralValidator(
            db=MissingSongDb(),
            model=EvalRaisesModel(),
            enabled=True,
            sample_rate=4,
            window_seconds=5.0,
        )

        result = validator.evaluate_top_candidates(
            np.ones(20, dtype=np.float32),
            [self.make_candidate()],
        )

        self.assertTrue(result.checked)
        self.assertEqual("shadow_wide", result.reason)
        self.assertEqual([], result.results)
        self.assertIsNone(result.error)

    def test_query_too_short_returns_unchecked_without_evaluating_model(self):
        validator = NeuralValidator(
            db=FakeDb(),
            model=EvalRaisesModel(),
            enabled=True,
            sample_rate=4,
            window_seconds=5.0,
        )

        result = validator.evaluate_top_candidates(
            np.ones(4, dtype=np.float32),
            [self.make_candidate()],
        )

        self.assertFalse(result.checked)
        self.assertEqual("query_too_short", result.reason)
        self.assertEqual([], result.results)
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
