# Neural Model Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the neural validator to symmetric torchaudio features, a stronger logits-based CNN, reproducible song splits, and a CLI-driven training module.

**Architecture:** Keep fingerprint recognition unchanged. Add a separate neural feature path based on torchaudio, with song-level split files consumed by training and evaluation. The runtime validator uses the same feature extractor and model contract as training, applying sigmoid outside the model.

**Tech Stack:** Python `unittest`, NumPy, PyTorch, torchaudio, existing CLI parser/handler, existing `MusicRecognitionService` and song database.

---

## File Structure

- Modify `requirements.txt`
  - Add `torchaudio`.
- Modify `config.py`
  - Add neural feature, split, and training defaults.
- Create `src/neural/splits.py`
  - Song split dataclasses, collect/split/save/load helpers.
- Modify `src/cli/parser.py`
  - Add `neural-split` and `neural-train` commands.
- Modify `src/cli/commands.py`
  - Dispatch `neural-split` and `neural-train`.
- Replace `src/neural/features.py`
  - Torchaudio symmetric feature extractor.
- Replace `src/neural/model.py`
  - Stronger logits-based pair classifier.
- Modify `src/neural/validator.py`
  - Use symmetric features, model logits, external sigmoid.
- Create `src/neural/pairs.py`
  - Pair sampling utilities for training/validation.
- Create `src/neural/training.py`
  - Training config, train/eval/checkpoint routines, CLI entrypoint helper.
- Add/update tests:
  - `tests/test_neural_splits.py`
  - `tests/test_parser.py`
  - `tests/test_neural_features.py`
  - `tests/test_neural_model.py`
  - `tests/test_neural_validator.py`
  - `tests/test_neural_pairs.py`
  - `tests/test_neural_training.py`

---

### Task 1: Config And Dependency Defaults

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py`
- Test: `tests/test_parser.py`

- [ ] **Step 1: Write the failing config defaults test**

Append to `tests/test_parser.py`:

```python
class NeuralConfigTests(unittest.TestCase):
    def test_neural_training_config_defaults_exist(self):
        self.assertEqual(80, cfg.NEURAL_N_MELS)
        self.assertEqual(384, cfg.NEURAL_MEL_HOP_LENGTH)
        self.assertEqual(1024, cfg.NEURAL_MEL_N_FFT)
        self.assertEqual("symmetric_mean_absdiff", cfg.NEURAL_INPUT_MODE)
        self.assertEqual(128, cfg.NEURAL_TRAIN_BATCH_SIZE)
        self.assertEqual(30, cfg.NEURAL_TRAIN_EPOCHS)
        self.assertEqual(1e-3, cfg.NEURAL_TRAIN_LR)
        self.assertEqual(1e-4, cfg.NEURAL_TRAIN_WEIGHT_DECAY)
        self.assertTrue(cfg.NEURAL_TRAIN_MIXED_PRECISION)
        self.assertEqual(4, cfg.NEURAL_TRAIN_NUM_WORKERS)
        self.assertEqual(0.80, cfg.NEURAL_SPLIT_TRAIN_RATIO)
        self.assertEqual(0.10, cfg.NEURAL_SPLIT_VALIDATION_RATIO)
        self.assertEqual(0.10, cfg.NEURAL_SPLIT_TEST_RATIO)
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
python -m unittest tests.test_parser
```

Expected: FAIL because the new config names do not exist or old neural defaults differ.

- [ ] **Step 3: Add dependency**

Add this line to `requirements.txt`:

```text
torchaudio
```

- [ ] **Step 4: Add config defaults**

Modify neural settings in `config.py`:

```python
NEURAL_N_MELS = int(os.getenv("NEURAL_N_MELS", "80"))
NEURAL_MEL_HOP_LENGTH = int(os.getenv("NEURAL_MEL_HOP_LENGTH", "384"))
NEURAL_MEL_N_FFT = int(os.getenv("NEURAL_MEL_N_FFT", str(N_FFT)))
NEURAL_INPUT_MODE = os.getenv("NEURAL_INPUT_MODE", "symmetric_mean_absdiff")
NEURAL_TRAIN_BATCH_SIZE = int(os.getenv("NEURAL_TRAIN_BATCH_SIZE", "128"))
NEURAL_TRAIN_EPOCHS = int(os.getenv("NEURAL_TRAIN_EPOCHS", "30"))
NEURAL_TRAIN_LR = float(os.getenv("NEURAL_TRAIN_LR", "1e-3"))
NEURAL_TRAIN_WEIGHT_DECAY = float(os.getenv("NEURAL_TRAIN_WEIGHT_DECAY", "1e-4"))
NEURAL_TRAIN_MIXED_PRECISION = os.getenv("NEURAL_TRAIN_MIXED_PRECISION", "True").lower() in ("true", "1", "yes")
NEURAL_TRAIN_NUM_WORKERS = int(os.getenv("NEURAL_TRAIN_NUM_WORKERS", "4"))
NEURAL_SPLIT_PATH = get_path("NEURAL_SPLIT_PATH", "data/neural/splits/song_split.json")
NEURAL_SPLIT_TRAIN_RATIO = float(os.getenv("NEURAL_SPLIT_TRAIN_RATIO", "0.80"))
NEURAL_SPLIT_VALIDATION_RATIO = float(os.getenv("NEURAL_SPLIT_VALIDATION_RATIO", "0.10"))
NEURAL_SPLIT_TEST_RATIO = float(os.getenv("NEURAL_SPLIT_TEST_RATIO", "0.10"))
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m unittest tests.test_parser
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add config.py requirements.txt tests/test_parser.py
git commit -m "Add neural training config defaults"
```

---

### Task 2: Song Split Module

**Files:**
- Create: `src/neural/splits.py`
- Test: `tests/test_neural_splits.py`

- [ ] **Step 1: Write failing split tests**

Create `tests/test_neural_splits.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from src.neural.splits import (
    SongSplit,
    SongSplitItem,
    collect_song_items,
    load_song_split,
    save_song_split,
    split_song_items,
)


class FakeSongs:
    def __init__(self, songs):
        self.db = {song["song_id"]: song for song in songs}


class FakeDb:
    def __init__(self, songs):
        self.songs = FakeSongs(songs)


class NeuralSplitTests(unittest.TestCase):
    def songs(self, count=10):
        return [
            {
                "song_id": i,
                "file_path": Path(f"data/processed/song_{i}.wav"),
                "title": f"Song {i}",
                "artist": "Artist",
                "duration": 100.0 + i,
            }
            for i in range(count)
        ]

    def test_collect_song_items_sorts_by_song_id(self):
        items = collect_song_items(FakeDb(list(reversed(self.songs(3)))))

        self.assertEqual([0, 1, 2], [item.song_id for item in items])

    def test_split_song_items_uses_80_10_10_counts(self):
        split = split_song_items(self.songs(10), seed=10, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1)

        self.assertEqual(8, len(split.train))
        self.assertEqual(1, len(split.validation_heldout))
        self.assertEqual(1, len(split.test_heldout))
        all_ids = [item.song_id for item in split.train + split.validation_heldout + split.test_heldout]
        self.assertEqual(sorted(all_ids), list(range(10)))

    def test_split_is_reproducible_for_same_seed(self):
        first = split_song_items(self.songs(20), seed=123, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1)
        second = split_song_items(self.songs(20), seed=123, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1)

        self.assertEqual([item.song_id for item in first.train], [item.song_id for item in second.train])

    def test_save_and_load_split_json(self):
        split = split_song_items(self.songs(10), seed=10, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            save_song_split(split, path)
            loaded = load_song_split(path)

        self.assertEqual(split.counts, loaded.counts)
        self.assertEqual([item.song_id for item in split.train], [item.song_id for item in loaded.train])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m unittest tests.test_neural_splits
```

Expected: FAIL because `src.neural.splits` does not exist.

- [ ] **Step 3: Implement split module**

Create `src/neural/splits.py`:

```python
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

import config as cfg


@dataclass
class SongSplitItem:
    song_id: int
    file_path: str
    title: str
    artist: str
    duration: float


@dataclass
class SongSplit:
    version: int
    seed: int
    ratios: dict[str, float]
    counts: dict[str, int]
    train: list[SongSplitItem]
    validation_heldout: list[SongSplitItem]
    test_heldout: list[SongSplitItem]


def collect_song_items(service_or_db) -> list[SongSplitItem]:
    db = getattr(service_or_db, "db", service_or_db)
    songs = getattr(db, "songs", db).db.values()
    return [
        song_item_from_dict(song)
        for song in sorted(songs, key=lambda item: int(item["song_id"]))
    ]


def song_item_from_dict(song: dict[str, Any]) -> SongSplitItem:
    return SongSplitItem(
        song_id=int(song["song_id"]),
        file_path=portable_path(song["file_path"]),
        title=str(song.get("title", "")),
        artist=str(song.get("artist", "")),
        duration=float(song.get("duration", 0.0)),
    )


def portable_path(path_value) -> str:
    path = Path(path_value)
    try:
        return path.resolve().relative_to(cfg.BASE_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def split_song_items(
    items,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> SongSplit:
    normalized = [item if isinstance(item, SongSplitItem) else song_item_from_dict(item) for item in items]
    normalized.sort(key=lambda item: item.song_id)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(normalized))
    shuffled = [normalized[int(index)] for index in order]

    total = len(shuffled)
    train_count = int(total * train_ratio)
    validation_count = int(total * validation_ratio)
    test_count = total - train_count - validation_count

    train = shuffled[:train_count]
    validation = shuffled[train_count:train_count + validation_count]
    test = shuffled[train_count + validation_count:]
    return SongSplit(
        version=1,
        seed=seed,
        ratios={
            "train": train_ratio,
            "validation_heldout": validation_ratio,
            "test_heldout": test_ratio,
        },
        counts={
            "total": total,
            "train": len(train),
            "validation_heldout": len(validation),
            "test_heldout": len(test),
        },
        train=train,
        validation_heldout=validation,
        test_heldout=test,
    )


def save_song_split(split: SongSplit, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(split), file, ensure_ascii=False, indent=2)


def load_song_split(path: Path) -> SongSplit:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return SongSplit(
        version=int(data["version"]),
        seed=int(data["seed"]),
        ratios={key: float(value) for key, value in data["ratios"].items()},
        counts={key: int(value) for key, value in data["counts"].items()},
        train=[SongSplitItem(**item) for item in data["train"]],
        validation_heldout=[SongSplitItem(**item) for item in data["validation_heldout"]],
        test_heldout=[SongSplitItem(**item) for item in data["test_heldout"]],
    )
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m unittest tests.test_neural_splits
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/neural/splits.py tests/test_neural_splits.py
git commit -m "Add neural song split helpers"
```

---

### Task 3: Neural Split CLI

**Files:**
- Modify: `src/cli/parser.py`
- Modify: `src/cli/commands.py`
- Test: `tests/test_parser.py`
- Test: `tests/test_neural_splits.py`

- [ ] **Step 1: Add parser test**

Append to `tests/test_parser.py`:

```python
class NeuralSplitParserTests(unittest.TestCase):
    def test_neural_split_command_accepts_split_flags(self):
        parser = build_parser()

        args = parser.parse_args([
            "neural-split",
            "--output",
            "data/neural/custom.json",
            "--seed",
            "123",
            "--train-ratio",
            "0.7",
            "--validation-ratio",
            "0.2",
            "--test-ratio",
            "0.1",
            "--force",
        ])

        self.assertEqual("neural-split", args.action)
        self.assertEqual("data\\neural\\custom.json", str(args.output))
        self.assertEqual(123, args.seed)
        self.assertEqual(0.7, args.train_ratio)
        self.assertEqual(0.2, args.validation_ratio)
        self.assertEqual(0.1, args.test_ratio)
        self.assertTrue(args.force)
```

- [ ] **Step 2: Add command handler tests**

Append to `tests/test_neural_splits.py`:

```python
from types import SimpleNamespace
from unittest.mock import patch

from src.cli.commands import CommandHandler


class NeuralSplitCommandTests(unittest.TestCase):
    def test_command_writes_split_file(self):
        service = SimpleNamespace(db=FakeDb(self.songs(10)))
        handler = CommandHandler(service)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "split.json"
            args = SimpleNamespace(
                output=output,
                seed=10,
                train_ratio=0.8,
                validation_ratio=0.1,
                test_ratio=0.1,
                force=False,
            )
            handler.neural_split(args)

            loaded = load_song_split(output)

        self.assertEqual(8, loaded.counts["train"])

    def test_command_refuses_to_overwrite_without_force(self):
        service = SimpleNamespace(db=FakeDb(self.songs(10)))
        handler = CommandHandler(service)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "split.json"
            output.write_text("existing", encoding="utf-8")
            args = SimpleNamespace(output=output, seed=10, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, force=False)

            with self.assertRaises(FileExistsError):
                handler.neural_split(args)
```

- [ ] **Step 3: Run failing tests**

Run:

```powershell
python -m unittest tests.test_parser tests.test_neural_splits
```

Expected: FAIL because parser/handler do not expose `neural-split`.

- [ ] **Step 4: Implement parser command**

Modify `src/cli/parser.py`:

```python
    parser_neural_split = subparsers.add_parser(
        "neural-split",
        help="Create a reproducible train/validation/test song split for neural training.",
    )
    parser_neural_split.add_argument("--output", type=Path, default=cfg.NEURAL_SPLIT_PATH)
    parser_neural_split.add_argument("--seed", type=int, default=cfg.RNG_SEED)
    parser_neural_split.add_argument("--train-ratio", type=float, default=cfg.NEURAL_SPLIT_TRAIN_RATIO)
    parser_neural_split.add_argument("--validation-ratio", type=float, default=cfg.NEURAL_SPLIT_VALIDATION_RATIO)
    parser_neural_split.add_argument("--test-ratio", type=float, default=cfg.NEURAL_SPLIT_TEST_RATIO)
    parser_neural_split.add_argument("--force", action="store_true")
```

- [ ] **Step 5: Implement command handler**

Modify imports in `src/cli/commands.py`:

```python
from src.neural.splits import collect_song_items, save_song_split, split_song_items
```

Add action mapping:

```python
            "neural-split": self.neural_split,
```

Add method:

```python
    def neural_split(self, args) -> None:
        output = args.output
        if output.exists() and not args.force:
            raise FileExistsError(f"Split file already exists: {output}. Use --force to overwrite.")

        songs = collect_song_items(self.service)
        if not songs:
            raise ValueError("No songs found in the database. Run process/recreate before neural-split.")

        split = split_song_items(
            songs,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
        )
        save_song_split(split, output)
        print(
            "Saved neural split to "
            f"{output}: train={split.counts['train']} "
            f"validation_heldout={split.counts['validation_heldout']} "
            f"test_heldout={split.counts['test_heldout']}"
        )
```

- [ ] **Step 6: Run tests**

Run:

```powershell
python -m unittest tests.test_parser tests.test_neural_splits
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src/cli/parser.py src/cli/commands.py tests/test_parser.py tests/test_neural_splits.py
git commit -m "Add neural split CLI"
```

---

### Task 4: Symmetric Torchaudio Feature Extractor

**Files:**
- Modify: `src/neural/features.py`
- Test: `tests/test_neural_features.py`

- [ ] **Step 1: Replace feature tests**

Update `tests/test_neural_features.py` with:

```python
import unittest

import numpy as np
import torch

from src.neural.features import TorchMelPairFeatureExtractor, build_symmetric_pair_features


class NeuralFeatureTests(unittest.TestCase):
    def test_symmetric_pair_features_are_invariant_to_swap(self):
        left = torch.randn(2, 80, 144)
        right = torch.randn(2, 80, 144)

        first = build_symmetric_pair_features(left, right)
        second = build_symmetric_pair_features(right, left)

        self.assertEqual((2, 2, 80, 144), tuple(first.shape))
        torch.testing.assert_close(first, second)

    def test_symmetric_pair_features_channels_are_mean_and_abs_diff(self):
        left = torch.tensor([[[1.0, 3.0]]])
        right = torch.tensor([[[5.0, 1.0]]])

        features = build_symmetric_pair_features(left, right)

        torch.testing.assert_close(features[:, 0], torch.tensor([[[3.0, 2.0]]]))
        torch.testing.assert_close(features[:, 1], torch.tensor([[[4.0, 2.0]]]))

    def test_torch_extractor_returns_two_channel_batch(self):
        extractor = TorchMelPairFeatureExtractor(sample_rate=11025, n_mels=32, n_fft=512, hop_length=256)
        left = torch.zeros(2, 11025)
        right = torch.ones(2, 11025)

        features = extractor(left, right)

        self.assertEqual(2, features.shape[0])
        self.assertEqual(2, features.shape[1])
        self.assertEqual(32, features.shape[2])
        self.assertTrue(torch.all(torch.isfinite(features)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m unittest tests.test_neural_features
```

Expected: FAIL because torchaudio extractor and symmetric functions do not exist.

- [ ] **Step 3: Implement torchaudio extractor**

Replace `src/neural/features.py` with:

```python
import torch
import torch.nn as nn
import torchaudio


class TorchMelPairFeatureExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, query_audio: torch.Tensor, candidate_audio: torch.Tensor) -> torch.Tensor:
        query = self.log_mel(query_audio)
        candidate = self.log_mel(candidate_audio)
        frame_count = min(query.shape[-1], candidate.shape[-1])
        if frame_count <= 0:
            raise ValueError("log-mel features must have at least one frame")
        return build_symmetric_pair_features(query[..., :frame_count], candidate[..., :frame_count])

    def log_mel(self, audio: torch.Tensor) -> torch.Tensor:
        if not torch.all(torch.isfinite(audio)):
            raise ValueError("audio must contain only finite values")
        values = self.to_db(self.mel(audio.float()))
        mean = values.mean(dim=(-2, -1), keepdim=True)
        std = values.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return (values - mean) / (std + 1e-6)


def build_symmetric_pair_features(query_mel: torch.Tensor, candidate_mel: torch.Tensor) -> torch.Tensor:
    mean = (query_mel + candidate_mel) * 0.5
    abs_diff = torch.abs(query_mel - candidate_mel)
    return torch.stack([mean, abs_diff], dim=1)
```

- [ ] **Step 4: Run feature tests**

Run:

```powershell
python -m unittest tests.test_neural_features
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/neural/features.py tests/test_neural_features.py
git commit -m "Use symmetric torchaudio pair features"
```

---

### Task 5: Stronger Logits Pair Classifier

**Files:**
- Modify: `src/neural/model.py`
- Test: `tests/test_neural_model.py`

- [ ] **Step 1: Replace model tests**

Update `tests/test_neural_model.py`:

```python
import unittest

import torch

from src.neural.model import PairClassifier


class PairClassifierTests(unittest.TestCase):
    def test_forward_returns_logits_per_pair(self):
        model = PairClassifier(input_channels=2)
        batch = torch.zeros((4, 2, 80, 144), dtype=torch.float32)

        logits = model(batch)

        self.assertEqual((4,), tuple(logits.shape))

    def test_model_does_not_apply_sigmoid(self):
        model = PairClassifier(input_channels=2)
        modules = [type(module).__name__ for module in model.modules()]

        self.assertNotIn("Sigmoid", modules)

    def test_forward_preserves_batch_dimension_for_single_pair_in_eval_mode(self):
        model = PairClassifier(input_channels=2)
        model.eval()

        logits = model(torch.zeros((1, 2, 80, 144), dtype=torch.float32))

        self.assertEqual((1,), tuple(logits.shape))

    def test_rejects_non_positive_input_channels(self):
        with self.assertRaisesRegex(ValueError, "input_channels must be positive"):
            PairClassifier(input_channels=0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m unittest tests.test_neural_model
```

Expected: FAIL because model still uses old architecture and Sigmoid.

- [ ] **Step 3: Implement stronger model**

Replace `src/neural/model.py` with:

```python
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class PairClassifier(nn.Module):
    def __init__(self, input_channels: int = 2) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")

        self.features = nn.Sequential(
            ConvBlock(input_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 192),
            ConvBlock(192, 192),
            nn.MaxPool2d(2),
            ConvBlock(192, 256),
            ConvBlock(256, 256),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.features(batch)
        pooled_avg = features.mean(dim=(-2, -1))
        pooled_max = features.amax(dim=(-2, -1))
        logits = self.head(torch.cat([pooled_avg, pooled_max], dim=1))
        return logits.squeeze(dim=1)
```

- [ ] **Step 4: Run model tests**

Run:

```powershell
python -m unittest tests.test_neural_model
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/neural/model.py tests/test_neural_model.py
git commit -m "Use logits pair classifier architecture"
```

---

### Task 6: Update Validator For Logits And Symmetric Features

**Files:**
- Modify: `src/neural/validator.py`
- Test: `tests/test_neural_validator.py`

- [ ] **Step 1: Update validator tests**

In `tests/test_neural_validator.py`, change fake model outputs from probabilities to logits:

```python
class FakeModel:
    def eval(self):
        return self

    def __call__(self, batch):
        return torch.tensor([2.0], dtype=torch.float32)
```

Add assertion:

```python
self.assertAlmostEqual(round(float(torch.sigmoid(torch.tensor(2.0)).item()), 6), result.results[0].same_probability)
```

Change low-probability fake model:

```python
class LowProbabilityModel:
    def eval(self):
        return self

    def __call__(self, batch):
        return torch.tensor([-2.0], dtype=torch.float32)
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m unittest tests.test_neural_validator
```

Expected: FAIL because validator treats model output as probability directly.

- [ ] **Step 3: Update validator**

Modify `src/neural/validator.py`:

```python
from .features import TorchMelPairFeatureExtractor
```

Initialize extractor in `NeuralValidator.__init__`:

```python
self.feature_extractor = TorchMelPairFeatureExtractor(
    sample_rate=sample_rate,
    n_mels=n_mels,
    n_fft=n_fft,
    hop_length=hop_length,
)
```

Use `PairClassifier(input_channels=2)` in `_load_model`.

Update `_predict_probability`:

```python
    def _predict_probability(self, model, query_window, candidate_window) -> float:
        query = torch.from_numpy(query_window).unsqueeze(0).float()
        candidate = torch.from_numpy(candidate_window).unsqueeze(0).float()
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(query, candidate)
            logits = model(features)
            probability = torch.sigmoid(logits)[0].item()
        return round(float(probability), 6)
```

- [ ] **Step 4: Run validator tests**

Run:

```powershell
python -m unittest tests.test_neural_validator tests.test_neural_features tests.test_neural_model
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/neural/validator.py tests/test_neural_validator.py
git commit -m "Use logits and symmetric features in validator"
```

---

### Task 7: Pair Sampling Utilities

**Files:**
- Create: `src/neural/pairs.py`
- Test: `tests/test_neural_pairs.py`

- [ ] **Step 1: Write failing pair sampling tests**

Create `tests/test_neural_pairs.py`:

```python
import unittest

import numpy as np

from src.neural.pairs import choose_query_valid_seconds, sample_pair_kind


class NeuralPairSamplingTests(unittest.TestCase):
    def test_query_duration_distribution_has_expected_buckets(self):
        rng = np.random.default_rng(1)
        values = [choose_query_valid_seconds(rng) for _ in range(200)]

        self.assertTrue(any(value == 5.0 for value in values))
        self.assertTrue(any(3.0 <= value < 5.0 for value in values))
        self.assertTrue(any(2.0 <= value < 3.0 for value in values))

    def test_sample_pair_kind_returns_positive_and_negative_kinds(self):
        rng = np.random.default_rng(2)
        kinds = {sample_pair_kind(rng, positive_ratio=0.5, hard_negative_ratio=0.2) for _ in range(200)}

        self.assertIn("positive_same_time", kinds)
        self.assertIn("positive_jittered", kinds)
        self.assertIn("negative_random", kinds)
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m unittest tests.test_neural_pairs
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement sampling helpers**

Create `src/neural/pairs.py`:

```python
import numpy as np


def choose_query_valid_seconds(rng, window_seconds: float = 5.0) -> float:
    value = float(rng.random())
    if value < 0.70:
        return window_seconds
    if value < 0.90:
        return float(rng.uniform(3.0, window_seconds))
    return float(rng.uniform(2.0, 3.0))


def sample_pair_kind(rng, positive_ratio: float, hard_negative_ratio: float) -> str:
    if float(rng.random()) < positive_ratio:
        return "positive_same_time" if float(rng.random()) < 0.70 else "positive_jittered"
    return "negative_hard" if float(rng.random()) < hard_negative_ratio else "negative_random"
```

- [ ] **Step 4: Run pair tests**

Run:

```powershell
python -m unittest tests.test_neural_pairs
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/neural/pairs.py tests/test_neural_pairs.py
git commit -m "Add neural pair sampling helpers"
```

---

### Task 8: Training Module Core

**Files:**
- Create: `src/neural/training.py`
- Test: `tests/test_neural_training.py`

- [ ] **Step 1: Write failing training tests**

Create `tests/test_neural_training.py`:

```python
import tempfile
import unittest
from pathlib import Path

import torch

from src.neural.training import (
    TrainingConfig,
    evaluate_logits,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


class NeuralTrainingTests(unittest.TestCase):
    def test_evaluate_logits_returns_probability_rows(self):
        logits = torch.tensor([2.0, -2.0])
        labels = torch.tensor([1.0, 0.0])
        rows = evaluate_logits(logits, labels, pair_type="negative_random", duration_bucket="5.0s")

        self.assertEqual(2, len(rows))
        self.assertEqual(1, rows[0]["label"])
        self.assertEqual("negative_random", rows[0]["pair_type"])
        self.assertGreater(rows[0]["probability"], 0.5)
        self.assertLess(rows[1]["probability"], 0.5)

    def test_checkpoint_roundtrip(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = TrainingConfig(epochs=1, batch_size=2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, epoch=3, config=config, best_metric=0.42)
            loaded = load_checkpoint(path, model, optimizer)

        self.assertEqual(3, loaded["epoch"])
        self.assertEqual(0.42, loaded["best_metric"])

    def test_train_one_epoch_updates_model_and_returns_loss(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        batches = [
            (
                torch.ones((2, 1, 2, 2), dtype=torch.float32),
                torch.tensor([1.0, 0.0], dtype=torch.float32),
            )
        ]
        before = [param.detach().clone() for param in model.parameters()]

        loss = train_one_epoch(
            model=model,
            batches=batches,
            optimizer=optimizer,
            device=torch.device("cpu"),
            mixed_precision=False,
        )

        self.assertGreater(loss, 0.0)
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
python -m unittest tests.test_neural_training
```

Expected: FAIL because `src.neural.training` does not exist.

- [ ] **Step 3: Implement training core**

Create `src/neural/training.py`:

```python
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

import config as cfg


@dataclass
class TrainingConfig:
    epochs: int = cfg.NEURAL_TRAIN_EPOCHS
    batch_size: int = cfg.NEURAL_TRAIN_BATCH_SIZE
    learning_rate: float = cfg.NEURAL_TRAIN_LR
    weight_decay: float = cfg.NEURAL_TRAIN_WEIGHT_DECAY
    mixed_precision: bool = cfg.NEURAL_TRAIN_MIXED_PRECISION
    num_workers: int = cfg.NEURAL_TRAIN_NUM_WORKERS
    model_path: Path = cfg.NEURAL_MODEL_PATH
    split_path: Path = cfg.NEURAL_SPLIT_PATH


def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor, pair_type: str, duration_bucket: str) -> list[dict]:
    probabilities = torch.sigmoid(logits.detach()).cpu()
    labels_cpu = labels.detach().cpu()
    rows = []
    for probability, label in zip(probabilities, labels_cpu):
        rows.append(
            {
                "probability": float(probability.item()),
                "label": int(label.item()),
                "pair_type": pair_type,
                "duration_bucket": duration_bucket,
            }
        )
    return rows


def train_one_epoch(
    model: torch.nn.Module,
    batches,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixed_precision: bool,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    batch_count = 0
    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for features, labels in batches:
        features = features.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu().item())
        batch_count += 1

    return total_loss / batch_count if batch_count else 0.0


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainingConfig,
    best_metric: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
```

- [ ] **Step 4: Run training tests**

Run:

```powershell
python -m unittest tests.test_neural_training
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/neural/training.py tests/test_neural_training.py
git commit -m "Add neural training core helpers"
```

---

### Task 9: Neural Train CLI Skeleton

**Files:**
- Modify: `src/cli/parser.py`
- Modify: `src/cli/commands.py`
- Modify: `src/neural/training.py`
- Test: `tests/test_parser.py`
- Test: `tests/test_neural_training.py`

- [ ] **Step 1: Add parser test**

Append to `tests/test_parser.py`:

```python
class NeuralTrainParserTests(unittest.TestCase):
    def test_neural_train_command_accepts_training_flags(self):
        parser = build_parser()

        args = parser.parse_args([
            "neural-train",
            "--split",
            "data/neural/splits/custom.json",
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--device",
            "cpu",
        ])

        self.assertEqual("neural-train", args.action)
        self.assertEqual("data\\neural\\splits\\custom.json", str(args.split))
        self.assertEqual(2, args.epochs)
        self.assertEqual(16, args.batch_size)
        self.assertEqual("cpu", args.device)
```

- [ ] **Step 2: Add command test**

Append to `tests/test_neural_training.py`:

```python
from types import SimpleNamespace
from unittest.mock import patch

from src.cli.commands import CommandHandler


class NeuralTrainCommandTests(unittest.TestCase):
    def test_neural_train_dispatches_to_training_module(self):
        handler = CommandHandler(service=object())
        args = SimpleNamespace(split=Path("split.json"), epochs=2, batch_size=16, device="cpu")

        with patch("src.cli.commands.run_training") as run:
            handler.neural_train(args)

        run.assert_called_once()
```

- [ ] **Step 3: Run failing tests**

Run:

```powershell
python -m unittest tests.test_parser tests.test_neural_training
```

Expected: FAIL because `neural-train` parser and command do not exist.

- [ ] **Step 4: Add training entrypoint helper**

Append to `src/neural/training.py`:

```python
def run_training(args) -> None:
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        split_path=args.split,
    )
    print(
        "Neural training configured: "
        f"split={config.split_path} epochs={config.epochs} batch_size={config.batch_size}"
    )
```

- [ ] **Step 5: Add parser command**

Modify `src/cli/parser.py`:

```python
    parser_neural_train = subparsers.add_parser(
        "neural-train",
        help="Train the neural pair validator.",
    )
    parser_neural_train.add_argument("--split", type=Path, default=cfg.NEURAL_SPLIT_PATH)
    parser_neural_train.add_argument("--epochs", type=int, default=cfg.NEURAL_TRAIN_EPOCHS)
    parser_neural_train.add_argument("--batch-size", type=int, default=cfg.NEURAL_TRAIN_BATCH_SIZE)
    parser_neural_train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
```

- [ ] **Step 6: Add command handler**

Modify imports in `src/cli/commands.py`:

```python
from src.neural.training import run_training
```

Add action mapping:

```python
            "neural-train": self.neural_train,
```

Add method:

```python
    def neural_train(self, args) -> None:
        run_training(args)
```

- [ ] **Step 7: Run tests**

Run:

```powershell
python -m unittest tests.test_parser tests.test_neural_training
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add src/cli/parser.py src/cli/commands.py src/neural/training.py tests/test_parser.py tests/test_neural_training.py
git commit -m "Add neural train CLI entrypoint"
```

---

### Task 10: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run focused neural tests**

Run:

```powershell
python -m unittest tests.test_neural_splits tests.test_neural_features tests.test_neural_model tests.test_neural_validator tests.test_neural_pairs tests.test_neural_training tests.test_parser
```

Expected: PASS.

- [ ] **Step 2: Run full discovery**

Run:

```powershell
python -m unittest discover tests
```

Expected: PASS.

- [ ] **Step 3: Compile**

Run:

```powershell
python -m compileall src tests
```

Expected: command exits with code 0.

- [ ] **Step 4: Inspect diff**

Run:

```powershell
git diff --stat
git diff -- config.py requirements.txt src/neural src/cli tests
```

Expected: diff only contains neural model/training/split work.

- [ ] **Step 5: Commit final fixes if needed**

If fixes were required:

```powershell
git add config.py requirements.txt src/neural src/cli tests
git commit -m "Stabilize neural model training implementation"
```

If no fixes were required, do not create an empty commit.

---

## Self-Review

- Spec coverage:
  - Torchaudio backend: Task 4.
  - Symmetric `[mean, abs_diff]` features: Task 4.
  - Stronger logits model: Task 5.
  - Validator sigmoid outside model: Task 6.
  - Song split module and JSON format: Task 2.
  - `neural-split` CLI: Task 3.
  - Pair sampling helpers: Task 7.
  - Training checkpoint/eval helpers: Task 8.
  - `neural-train` CLI entrypoint: Task 9.
  - Verification: Task 10.
- Intentional scope boundary:
  - This plan adds a working training epoch function, checkpointing, evaluation rows, and CLI entrypoint. A fully optimized real-audio DataLoader can be expanded after this foundation is verified.
  - Hard-negative persistence remains a follow-up fed by failure logs.
- Type consistency:
  - `SongSplitItem` and `SongSplit` are defined in Task 2 and used by CLI/training.
  - `TorchMelPairFeatureExtractor` is defined in Task 4 and used by validator.
  - `PairClassifier` defaults to `input_channels=2` in Task 5 and validator uses the same contract in Task 6.
