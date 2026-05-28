# Neural Shadow Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch neural pair-classifier shadow validator that records top-N candidate diagnostics without changing fingerprint search results.

**Architecture:** Keep `QueryPipeline` fingerprint-only and extend it to record top candidates in `SearchTrace`. Add a focused `src/neural` package for audio windowing, model input construction, model inference, dataset generation, and reports. `MusicRecognitionService` orchestrates optional shadow validation after normal search and always returns the original fingerprint result.

**Tech Stack:** Python `unittest`, NumPy, librosa, PyTorch, existing `config.py`, existing `src.processing.preprocess.load_audio`.

---

## File Structure

- Modify `config.py`
  - Add neural shadow settings: enable flag, top-N, 5-second window, threshold, model path, mel parameters.
- Modify `src/recognition/search_trace.py`
  - Add dataclasses for `CandidateTrace` and `NeuralCandidateTrace`.
  - Add neural and top-candidate fields to `SearchTrace`.
- Modify `src/recognition/query_pipeline.py`
  - Build `top_candidates` after time coherency using the same scoring formula as selection.
- Create `src/neural/__init__.py`
  - Export no heavy dependencies at import time.
- Create `src/neural/audio_windows.py`
  - Crop/pad query and candidate audio to fixed 5-second windows and compute reliability.
- Create `src/neural/features.py`
  - Convert windows into `[query_logmel, candidate_logmel, abs_diff]`.
- Create `src/neural/model.py`
  - Define the small pair-classifier CNN.
- Create `src/neural/validator.py`
  - Load model lazily, evaluate top candidates, and return trace-compatible results.
- Create `src/neural/dataset.py`
  - Generate labeled pair metadata and tensors for training/evaluation.
- Create `src/neural/evaluation.py`
  - Compute confusion matrices and grouped reports.
- Modify `src/recognition/service.py`
  - Add optional neural validator injection and shadow orchestration.
- Add tests:
  - `tests/test_top_candidates_trace.py`
  - `tests/test_neural_audio_windows.py`
  - `tests/test_neural_features.py`
  - `tests/test_neural_model.py`
  - `tests/test_neural_validator.py`
  - `tests/test_neural_dataset.py`
  - `tests/test_neural_evaluation.py`
  - `tests/test_neural_shadow_service.py`

---

### Task 1: Add Trace Dataclasses And Defaults

**Files:**
- Modify: `src/recognition/search_trace.py`
- Test: `tests/test_top_candidates_trace.py`

- [ ] **Step 1: Write the failing trace defaults test**

Create `tests/test_top_candidates_trace.py`:

```python
import unittest

from src.recognition.search_trace import CandidateTrace, NeuralCandidateTrace, SearchTrace


class SearchTraceCandidateTests(unittest.TestCase):
    def test_search_trace_initializes_candidate_and_neural_lists(self):
        trace = SearchTrace()

        self.assertEqual([], trace.top_candidates)
        self.assertFalse(trace.neural_enabled)
        self.assertFalse(trace.neural_checked)
        self.assertEqual("", trace.neural_reason)
        self.assertEqual([], trace.neural_results)
        self.assertIsNone(trace.neural_error)

    def test_candidate_trace_has_fingerprint_fields(self):
        candidate = CandidateTrace(
            song_id=7,
            rank=1,
            score=0.25,
            max_count=10,
            time_offset_bins=-3,
            time_offset_seconds=0.209,
        )

        self.assertEqual(7, candidate.song_id)
        self.assertEqual(1, candidate.rank)
        self.assertEqual(0.25, candidate.score)
        self.assertEqual(10, candidate.max_count)
        self.assertEqual(-3, candidate.time_offset_bins)
        self.assertEqual(0.209, candidate.time_offset_seconds)

    def test_neural_candidate_trace_has_shadow_fields(self):
        result = NeuralCandidateTrace(
            song_id=7,
            rank=1,
            fingerprint_score=0.25,
            fingerprint_max_count=10,
            fingerprint_time_offset_seconds=0.209,
            same_probability=0.82,
            decision="same",
            threshold=0.7,
            reliability="high",
            query_valid_seconds=5.0,
            candidate_valid_seconds=5.0,
            padding_ratio=0.0,
        )

        self.assertEqual("same", result.decision)
        self.assertEqual(0.82, result.same_probability)
        self.assertEqual("high", result.reliability)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
python -m unittest tests.test_top_candidates_trace
```

Expected: FAIL with an import error for `CandidateTrace` or missing `SearchTrace` fields.

- [ ] **Step 3: Implement trace dataclasses and defaults**

Modify `src/recognition/search_trace.py`:

```python
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CandidateTrace:
    song_id: int
    rank: int
    score: float
    max_count: int
    time_offset_bins: int
    time_offset_seconds: float


@dataclass
class NeuralCandidateTrace:
    song_id: int
    rank: int
    fingerprint_score: float
    fingerprint_max_count: int
    fingerprint_time_offset_seconds: float
    same_probability: float
    decision: str
    threshold: float
    reliability: str
    query_valid_seconds: float
    candidate_valid_seconds: float
    padding_ratio: float


@dataclass
class SearchTrace:
    expected_id: int = -1
    query_fp_count: int = 0

    db_match_count: int = 0
    correct_in_db_lookup: bool = False
    raw_matches_by_song: Dict[int, int] = None
    expected_raw_match_count: int = 0

    candidates_after_filter: list[int] = None
    correct_after_filter: bool = False
    unique_addresses_by_song: Dict[int, int] = None
    expected_unique_address_count: int = 0

    candidates_after_time: Dict[int, tuple[int, int]] = None
    correct_after_time: bool = False
    correct_time_result: tuple[int, int] | None = None
    expected_offset_buckets: Dict[int, int] = None
    top_candidates: list[CandidateTrace] = None

    selected_id: int = -1
    selected_score: float = 0.0
    expected_score: float | None = None
    expected_time_offset: int | None = None
    selected_max_count: int = 0
    expected_max_count: int | None = None
    failure_analysis: Any = None
    attempt_sample_offset: int = 0
    offset_fallback_selected: bool = False
    offset_fallback_attempts: list[dict[str, Any]] = None

    neural_enabled: bool = False
    neural_checked: bool = False
    neural_reason: str = ""
    neural_results: list[NeuralCandidateTrace] = None
    neural_error: str | None = None

    dropped_stage: str = "unknown"
    reason: str = ""

    def __post_init__(self):
        if self.candidates_after_filter is None:
            self.candidates_after_filter = []
        if self.candidates_after_time is None:
            self.candidates_after_time = {}
        if self.raw_matches_by_song is None:
            self.raw_matches_by_song = {}
        if self.unique_addresses_by_song is None:
            self.unique_addresses_by_song = {}
        if self.expected_offset_buckets is None:
            self.expected_offset_buckets = {}
        if self.offset_fallback_attempts is None:
            self.offset_fallback_attempts = []
        if self.top_candidates is None:
            self.top_candidates = []
        if self.neural_results is None:
            self.neural_results = []

    @property
    def total_matches(self) -> int:
        return self.db_match_count
```

- [ ] **Step 4: Run the trace test**

Run:

```powershell
python -m unittest tests.test_top_candidates_trace
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src\recognition\search_trace.py tests\test_top_candidates_trace.py
git commit -m "Add neural candidate trace fields"
```

---

### Task 2: Record Top-N Fingerprint Candidates

**Files:**
- Modify: `config.py`
- Modify: `src/recognition/query_pipeline.py`
- Test: `tests/test_top_candidates_trace.py`

- [ ] **Step 1: Add failing tests for candidate ranking**

Append to `tests/test_top_candidates_trace.py`:

```python
from unittest.mock import patch

import config as cfg
from src.recognition.query_pipeline import build_top_candidates


class TopCandidateBuilderTests(unittest.TestCase):
    def test_build_top_candidates_sorts_by_score_then_max_count(self):
        results = {
            10: (2, -1),
            20: (6, 2),
            30: (4, 3),
        }

        with patch.object(cfg, "NEURAL_SHADOW_TOP_N", 2, create=True):
            candidates = build_top_candidates(results, total_matches=20)

        self.assertEqual([20, 30], [candidate.song_id for candidate in candidates])
        self.assertEqual([1, 2], [candidate.rank for candidate in candidates])
        self.assertAlmostEqual(6 / 20, candidates[0].score)
        self.assertEqual(6, candidates[0].max_count)
        self.assertEqual(2, candidates[0].time_offset_bins)
        self.assertAlmostEqual(-cfg.BIN_TIME * 2, candidates[0].time_offset_seconds)

    def test_build_top_candidates_returns_empty_when_total_matches_is_zero(self):
        candidates = build_top_candidates({1: (4, 0)}, total_matches=0)

        self.assertEqual([], candidates)
```

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
python -m unittest tests.test_top_candidates_trace
```

Expected: FAIL because `build_top_candidates` does not exist.

- [ ] **Step 3: Add config defaults**

Modify `config.py`:

```python
NEURAL_SHADOW_ENABLED = os.getenv("NEURAL_SHADOW_ENABLED", "False").lower() in ("true", "1", "yes")
NEURAL_SHADOW_TOP_N = int(os.getenv("NEURAL_SHADOW_TOP_N", "3"))
NEURAL_WINDOW_SECONDS = float(os.getenv("NEURAL_WINDOW_SECONDS", "5.0"))
NEURAL_MIN_QUERY_SECONDS = float(os.getenv("NEURAL_MIN_QUERY_SECONDS", "2.0"))
NEURAL_DECISION_THRESHOLD = float(os.getenv("NEURAL_DECISION_THRESHOLD", "0.70"))
NEURAL_MODEL_PATH = get_path("NEURAL_MODEL_PATH", "data/models/neural_pair_classifier.pt")
NEURAL_N_MELS = int(os.getenv("NEURAL_N_MELS", "64"))
NEURAL_MEL_HOP_LENGTH = int(os.getenv("NEURAL_MEL_HOP_LENGTH", str(HOP_LENGTH)))
NEURAL_MEL_N_FFT = int(os.getenv("NEURAL_MEL_N_FFT", str(N_FFT)))
```

- [ ] **Step 4: Implement candidate builder and attach it in search**

Modify imports in `src/recognition/query_pipeline.py`:

```python
from .search_trace import CandidateTrace, SearchTrace
```

After `trace.correct_time_result = results.get(expected_id)`, add:

```python
        trace.top_candidates = build_top_candidates(results, total_matches)
```

Add this helper near `update_expected_trace`:

```python
def build_top_candidates(
    results: dict[int, tuple[int, int]],
    total_matches: int,
) -> list[CandidateTrace]:
    if not results or total_matches <= 0:
        return []

    ranked: list[tuple[float, int, int, int]] = []
    for song_id, (max_count, time_offset_bins) in results.items():
        score = compute_candidate_score(max_count, total_matches)
        ranked.append((score, max_count, song_id, time_offset_bins))

    ranked.sort(reverse=True)

    candidates: list[CandidateTrace] = []
    for rank, (score, max_count, song_id, time_offset_bins) in enumerate(
        ranked[:cfg.NEURAL_SHADOW_TOP_N],
        start=1,
    ):
        candidates.append(
            CandidateTrace(
                song_id=song_id,
                rank=rank,
                score=score,
                max_count=max_count,
                time_offset_bins=time_offset_bins,
                time_offset_seconds=-cfg.BIN_TIME * time_offset_bins,
            )
        )
    return candidates
```

- [ ] **Step 5: Run candidate tests**

Run:

```powershell
python -m unittest tests.test_top_candidates_trace
```

Expected: PASS.

- [ ] **Step 6: Run existing recognition tests**

Run:

```powershell
python -m unittest tests.test_search_offset_fallback
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```powershell
git add config.py src\recognition\query_pipeline.py tests\test_top_candidates_trace.py
git commit -m "Record top fingerprint candidates in trace"
```

---

### Task 3: Implement Neural Audio Windowing

**Files:**
- Create: `src/neural/__init__.py`
- Create: `src/neural/audio_windows.py`
- Test: `tests/test_neural_audio_windows.py`

- [ ] **Step 1: Write failing audio window tests**

Create `tests/test_neural_audio_windows.py`:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing audio window tests**

Run:

```powershell
python -m unittest tests.test_neural_audio_windows
```

Expected: FAIL because `src.neural.audio_windows` does not exist.

- [ ] **Step 3: Create package init**

Create `src/neural/__init__.py`:

```python
"""Neural validation helpers for recognition shadow mode."""
```

- [ ] **Step 4: Implement audio windowing**

Create `src/neural/audio_windows.py`:

```python
from dataclasses import dataclass

import numpy as np


@dataclass
class WindowMetadata:
    valid_seconds: float
    padding_seconds: float
    padding_ratio: float
    reliability: str
    skipped: bool = False


def prepare_query_window(
    audio,
    sample_rate: int,
    window_seconds: float,
    min_query_seconds: float = 2.0,
) -> tuple[np.ndarray, WindowMetadata]:
    audio_array = np.asarray(audio, dtype=np.float32)
    target_samples = int(round(window_seconds * sample_rate))
    valid_samples = min(len(audio_array), target_samples)
    valid_seconds = valid_samples / sample_rate if sample_rate else 0.0

    window = np.zeros(target_samples, dtype=np.float32)
    if valid_samples:
        window[:valid_samples] = audio_array[:valid_samples]

    padding_seconds = max(0.0, window_seconds - valid_seconds)
    padding_ratio = padding_seconds / window_seconds if window_seconds else 0.0
    reliability = reliability_for_query_duration(valid_seconds, min_query_seconds)

    return window, WindowMetadata(
        valid_seconds=valid_seconds,
        padding_seconds=padding_seconds,
        padding_ratio=padding_ratio,
        reliability=reliability,
        skipped=reliability == "skipped",
    )


def reliability_for_query_duration(
    valid_seconds: float,
    min_query_seconds: float = 2.0,
) -> str:
    if valid_seconds < min_query_seconds:
        return "skipped"
    if valid_seconds < 3.0:
        return "low"
    if valid_seconds < 5.0:
        return "medium"
    return "high"


def crop_candidate_window(
    audio,
    sample_rate: int,
    start_seconds: float,
    window_seconds: float,
) -> tuple[np.ndarray, WindowMetadata]:
    audio_array = np.asarray(audio, dtype=np.float32)
    target_samples = int(round(window_seconds * sample_rate))
    start_sample = int(round(start_seconds * sample_rate))
    end_sample = start_sample + target_samples

    source_start = max(0, start_sample)
    source_end = min(len(audio_array), end_sample)
    valid_samples = max(0, source_end - source_start)

    window = np.zeros(target_samples, dtype=np.float32)
    destination_start = max(0, -start_sample)
    if valid_samples:
        window[destination_start:destination_start + valid_samples] = audio_array[source_start:source_end]

    valid_seconds = valid_samples / sample_rate if sample_rate else 0.0
    padding_seconds = max(0.0, window_seconds - valid_seconds)
    padding_ratio = padding_seconds / window_seconds if window_seconds else 0.0

    return window, WindowMetadata(
        valid_seconds=valid_seconds,
        padding_seconds=padding_seconds,
        padding_ratio=padding_ratio,
        reliability="high" if valid_samples else "skipped",
        skipped=valid_samples == 0,
    )
```

- [ ] **Step 5: Run audio window tests**

Run:

```powershell
python -m unittest tests.test_neural_audio_windows
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```powershell
git add src\neural\__init__.py src\neural\audio_windows.py tests\test_neural_audio_windows.py
git commit -m "Add neural audio window preparation"
```

---

### Task 4: Build Log-Mel Pair Features

**Files:**
- Create: `src/neural/features.py`
- Test: `tests/test_neural_features.py`

- [ ] **Step 1: Write failing feature tests**

Create `tests/test_neural_features.py`:

```python
import unittest

import numpy as np

from src.neural.features import build_pair_features


class NeuralFeatureTests(unittest.TestCase):
    def test_build_pair_features_returns_three_matching_channels(self):
        query = np.sin(np.linspace(0, 1, 11025, dtype=np.float32))
        candidate = np.cos(np.linspace(0, 1, 11025, dtype=np.float32))

        features = build_pair_features(
            query,
            candidate,
            sample_rate=11025,
            n_mels=32,
            n_fft=512,
            hop_length=256,
        )

        self.assertEqual(3, features.shape[0])
        self.assertEqual(32, features.shape[1])
        self.assertEqual(features.shape[1:], features[2].shape)
        np.testing.assert_allclose(features[2], np.abs(features[0] - features[1]), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing feature tests**

Run:

```powershell
python -m unittest tests.test_neural_features
```

Expected: FAIL because `src.neural.features` does not exist.

- [ ] **Step 3: Implement feature extraction**

Create `src/neural/features.py`:

```python
import librosa
import numpy as np


def build_pair_features(
    query_audio,
    candidate_audio,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    query_mel = log_mel(query_audio, sample_rate, n_mels, n_fft, hop_length)
    candidate_mel = log_mel(candidate_audio, sample_rate, n_mels, n_fft, hop_length)
    diff = np.abs(query_mel - candidate_mel)
    return np.stack([query_mel, candidate_mel, diff]).astype(np.float32, copy=False)


def log_mel(
    audio,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    audio_array = np.asarray(audio, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_values = librosa.power_to_db(mel, ref=np.max)
    mean = float(log_values.mean())
    std = float(log_values.std())
    return ((log_values - mean) / (std + 1e-6)).astype(np.float32, copy=False)
```

- [ ] **Step 4: Run feature tests**

Run:

```powershell
python -m unittest tests.test_neural_features
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src\neural\features.py tests\test_neural_features.py
git commit -m "Add neural pair feature extraction"
```

---

### Task 5: Add Pair Classifier Model

**Files:**
- Create: `src/neural/model.py`
- Test: `tests/test_neural_model.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_neural_model.py`:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing model tests**

Run:

```powershell
python -m unittest tests.test_neural_model
```

Expected: FAIL because `src.neural.model` does not exist.

- [ ] **Step 3: Implement the model**

Create `src/neural/model.py`:

```python
import torch
import torch.nn as nn


class PairClassifier(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        values = self.features(batch)
        probabilities = self.classifier(values)
        return probabilities.squeeze(dim=1)
```

- [ ] **Step 4: Run model tests**

Run:

```powershell
python -m unittest tests.test_neural_model
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```powershell
git add src\neural\model.py tests\test_neural_model.py
git commit -m "Add neural pair classifier model"
```

---

### Task 6: Implement Neural Validator

**Files:**
- Create: `src/neural/validator.py`
- Test: `tests/test_neural_validator.py`

- [ ] **Step 1: Write failing validator tests**

Create `tests/test_neural_validator.py`:

```python
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.neural.validator import NeuralValidationResult, NeuralValidator
from src.recognition.search_trace import CandidateTrace


class FakeModel:
    def eval(self):
        return self

    def __call__(self, batch):
        import torch
        return torch.tensor([0.8], dtype=torch.float32)


class FakeDb:
    def get_song_by_id(self, song_id):
        return {"song_id": song_id, "file_path": Path("candidate.wav")}


class NeuralValidatorTests(unittest.TestCase):
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
        candidate = CandidateTrace(
            song_id=1,
            rank=1,
            score=0.12,
            max_count=6,
            time_offset_bins=0,
            time_offset_seconds=0.0,
        )

        with patch("src.neural.validator.pp.load_audio", return_value=np.ones(20, dtype=np.float32)):
            result = validator.evaluate_top_candidates(query, [candidate])

        self.assertTrue(result.checked)
        self.assertIsNone(result.error)
        self.assertEqual(1, len(result.results))
        self.assertEqual("same", result.results[0].decision)
        self.assertEqual(0.8, result.results[0].same_probability)

    def test_disabled_validator_returns_unchecked_result(self):
        validator = NeuralValidator(db=FakeDb(), model=FakeModel(), enabled=False)

        result = validator.evaluate_top_candidates(np.ones(20, dtype=np.float32), [])

        self.assertFalse(result.checked)
        self.assertEqual([], result.results)
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing validator tests**

Run:

```powershell
python -m unittest tests.test_neural_validator
```

Expected: FAIL because `src.neural.validator` does not exist.

- [ ] **Step 3: Implement validator**

Create `src/neural/validator.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import config as cfg
from src.processing import preprocess as pp
from src.recognition.search_trace import CandidateTrace, NeuralCandidateTrace
from .audio_windows import crop_candidate_window, prepare_query_window
from .features import build_pair_features
from .model import PairClassifier


@dataclass
class NeuralValidationResult:
    checked: bool
    reason: str
    results: list[NeuralCandidateTrace]
    error: str | None = None


class NeuralValidator:
    def __init__(
        self,
        db,
        model=None,
        enabled: bool = cfg.NEURAL_SHADOW_ENABLED,
        threshold: float = cfg.NEURAL_DECISION_THRESHOLD,
        top_n: int = cfg.NEURAL_SHADOW_TOP_N,
        sample_rate: int = cfg.SAMPLE_RATE,
        window_seconds: float = cfg.NEURAL_WINDOW_SECONDS,
        min_query_seconds: float = cfg.NEURAL_MIN_QUERY_SECONDS,
        n_mels: int = cfg.NEURAL_N_MELS,
        n_fft: int = cfg.NEURAL_MEL_N_FFT,
        hop_length: int = cfg.NEURAL_MEL_HOP_LENGTH,
        model_path: Path = cfg.NEURAL_MODEL_PATH,
    ) -> None:
        self.db = db
        self.model = model
        self.enabled = enabled
        self.threshold = threshold
        self.top_n = top_n
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.min_query_seconds = min_query_seconds
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model_path = Path(model_path)

    def evaluate_top_candidates(
        self,
        query_audio,
        candidates: list[CandidateTrace],
    ) -> NeuralValidationResult:
        if not self.enabled:
            return NeuralValidationResult(False, "disabled", [])
        if not candidates:
            return NeuralValidationResult(False, "no_candidates", [])

        try:
            model = self._load_model()
            query_window, query_meta = prepare_query_window(
                query_audio,
                sample_rate=self.sample_rate,
                window_seconds=self.window_seconds,
                min_query_seconds=self.min_query_seconds,
            )
            if query_meta.skipped:
                return NeuralValidationResult(False, "query_too_short", [])

            results: list[NeuralCandidateTrace] = []
            for candidate in candidates[:self.top_n]:
                song = self.db.get_song_by_id(candidate.song_id)
                if not song:
                    continue

                candidate_audio = pp.load_audio(Path(song["file_path"]), target_sr=self.sample_rate)
                candidate_window, candidate_meta = crop_candidate_window(
                    candidate_audio,
                    sample_rate=self.sample_rate,
                    start_seconds=candidate.time_offset_seconds,
                    window_seconds=self.window_seconds,
                )
                if candidate_meta.skipped:
                    continue

                probability = self._predict_probability(model, query_window, candidate_window)
                decision = "same" if probability >= self.threshold else "not_same"
                results.append(
                    NeuralCandidateTrace(
                        song_id=candidate.song_id,
                        rank=candidate.rank,
                        fingerprint_score=candidate.score,
                        fingerprint_max_count=candidate.max_count,
                        fingerprint_time_offset_seconds=candidate.time_offset_seconds,
                        same_probability=probability,
                        decision=decision,
                        threshold=self.threshold,
                        reliability=query_meta.reliability,
                        query_valid_seconds=query_meta.valid_seconds,
                        candidate_valid_seconds=candidate_meta.valid_seconds,
                        padding_ratio=query_meta.padding_ratio,
                    )
                )

            return NeuralValidationResult(True, "shadow_wide", results)
        except Exception as error:
            return NeuralValidationResult(True, "shadow_wide", [], error=str(error))

    def _load_model(self):
        if self.model is not None:
            self.model.eval()
            return self.model
        model = PairClassifier(input_channels=3)
        state = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        self.model = model
        return model

    def _predict_probability(self, model, query_window, candidate_window) -> float:
        features = build_pair_features(
            query_window,
            candidate_window,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        batch = torch.from_numpy(features).unsqueeze(0).float()
        with torch.no_grad():
            probability = model(batch)[0].item()
        return round(float(probability), 6)
```

- [ ] **Step 4: Run validator tests**

Run:

```powershell
python -m unittest tests.test_neural_validator
```

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

```powershell
git add src\neural\validator.py tests\test_neural_validator.py
git commit -m "Add neural shadow validator"
```

---

### Task 7: Integrate Shadow Validation Into Service

**Files:**
- Modify: `src/recognition/service.py`
- Test: `tests/test_neural_shadow_service.py`

- [ ] **Step 1: Write failing service integration tests**

Create `tests/test_neural_shadow_service.py`:

```python
import unittest

import numpy as np

from src.recognition.query_pipeline import SearchResult
from src.recognition.search_trace import CandidateTrace, NeuralCandidateTrace, SearchTrace
from src.recognition.service import MusicRecognitionService


class FakePipeline:
    def search(self, audio, expected_id=-1, file_path=None):
        trace = SearchTrace()
        trace.top_candidates = [
            CandidateTrace(
                song_id=1,
                rank=1,
                score=0.2,
                max_count=5,
                time_offset_bins=0,
                time_offset_seconds=0.0,
            )
        ]
        trace.selected_id = 1
        trace.selected_score = 0.2
        trace.selected_max_count = 5
        return SearchResult(song_id=1, time_offset=0.0, trace=trace)


class FakeValidator:
    def evaluate_top_candidates(self, audio, candidates):
        return type(
            "Result",
            (),
            {
                "checked": True,
                "reason": "shadow_wide",
                "error": None,
                "results": [
                    NeuralCandidateTrace(
                        song_id=1,
                        rank=1,
                        fingerprint_score=0.2,
                        fingerprint_max_count=5,
                        fingerprint_time_offset_seconds=0.0,
                        same_probability=0.9,
                        decision="same",
                        threshold=0.7,
                        reliability="high",
                        query_valid_seconds=5.0,
                        candidate_valid_seconds=5.0,
                        padding_ratio=0.0,
                    )
                ],
            },
        )()


class NeuralShadowServiceTests(unittest.TestCase):
    def make_service(self, validator):
        service = MusicRecognitionService.__new__(MusicRecognitionService)
        service.query_pipeline = FakePipeline()
        service.neural_validator = validator
        service.last_search_trace = None
        return service

    def test_shadow_validation_preserves_fingerprint_result(self):
        service = self.make_service(FakeValidator())

        song_id, time_offset = service.search_song(np.ones(30, dtype=np.float32), offset_fallback=False)

        self.assertEqual(1, song_id)
        self.assertEqual(0.0, time_offset)
        self.assertTrue(service.last_search_trace.neural_enabled)
        self.assertTrue(service.last_search_trace.neural_checked)
        self.assertEqual("shadow_wide", service.last_search_trace.neural_reason)
        self.assertEqual(1, len(service.last_search_trace.neural_results))

    def test_missing_validator_leaves_trace_unchecked(self):
        service = self.make_service(None)

        song_id, _ = service.search_song(np.ones(30, dtype=np.float32), offset_fallback=False)

        self.assertEqual(1, song_id)
        self.assertFalse(service.last_search_trace.neural_enabled)
        self.assertFalse(service.last_search_trace.neural_checked)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing service tests**

Run:

```powershell
python -m unittest tests.test_neural_shadow_service
```

Expected: FAIL because `MusicRecognitionService` does not orchestrate `neural_validator`.

- [ ] **Step 3: Modify service initialization and search orchestration**

Modify `src/recognition/service.py` imports:

```python
try:
    from src.neural.validator import NeuralValidator
except Exception:
    NeuralValidator = None
```

Modify `__init__` after `self.query_pipeline = QueryPipeline(self.db)`:

```python
        self.neural_validator = (
            NeuralValidator(self.db)
            if NeuralValidator is not None and cfg.NEURAL_SHADOW_ENABLED
            else None
        )
```

Modify `search_song()` before `self.last_search_trace = result.trace`:

```python
        self._run_neural_shadow_validation(audio, result.trace)
```

Add this method to `MusicRecognitionService`:

```python
    def _run_neural_shadow_validation(self, audio, trace: SearchTrace) -> None:
        validator = getattr(self, "neural_validator", None)
        if validator is None:
            trace.neural_enabled = False
            return

        trace.neural_enabled = True
        validation = validator.evaluate_top_candidates(audio, trace.top_candidates)
        trace.neural_checked = validation.checked
        trace.neural_reason = validation.reason
        trace.neural_results = validation.results
        trace.neural_error = validation.error
```

- [ ] **Step 4: Run service tests**

Run:

```powershell
python -m unittest tests.test_neural_shadow_service
```

Expected: PASS.

- [ ] **Step 5: Run fallback tests**

Run:

```powershell
python -m unittest tests.test_search_offset_fallback
```

Expected: PASS.

- [ ] **Step 6: Commit Task 7**

```powershell
git add src\recognition\service.py tests\test_neural_shadow_service.py
git commit -m "Run neural validator in shadow mode"
```

---

### Task 8: Add Dataset Pair Generation

**Files:**
- Create: `src/neural/dataset.py`
- Test: `tests/test_neural_dataset.py`

- [ ] **Step 1: Write failing dataset tests**

Create `tests/test_neural_dataset.py`:

```python
import unittest

from src.neural.dataset import PairExample, duration_bucket, make_random_negative, make_same_time_positive


SONGS = [
    {"song_id": 1, "file_path": "a.wav", "duration": 20.0},
    {"song_id": 2, "file_path": "b.wav", "duration": 30.0},
]


class NeuralDatasetTests(unittest.TestCase):
    def test_make_same_time_positive_uses_same_song_and_label_one(self):
        example = make_same_time_positive(SONGS[0], start_seconds=4.0, query_valid_seconds=5.0)

        self.assertEqual(1, example.label)
        self.assertEqual(1, example.query_song_id)
        self.assertEqual(1, example.candidate_song_id)
        self.assertEqual("positive_same_time", example.pair_type)
        self.assertEqual(0.0, example.padding_ratio)

    def test_make_random_negative_uses_different_songs_and_label_zero(self):
        example = make_random_negative(
            query_song=SONGS[0],
            candidate_song=SONGS[1],
            query_start_seconds=2.0,
            candidate_start_seconds=3.0,
            query_valid_seconds=3.0,
        )

        self.assertEqual(0, example.label)
        self.assertEqual(1, example.query_song_id)
        self.assertEqual(2, example.candidate_song_id)
        self.assertEqual("negative_random", example.pair_type)
        self.assertAlmostEqual(0.4, example.padding_ratio)

    def test_duration_bucket_names_query_duration_ranges(self):
        self.assertEqual("5.0s", duration_bucket(5.0))
        self.assertEqual("4-5s", duration_bucket(4.2))
        self.assertEqual("3-4s", duration_bucket(3.1))
        self.assertEqual("2-3s", duration_bucket(2.5))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing dataset tests**

Run:

```powershell
python -m unittest tests.test_neural_dataset
```

Expected: FAIL because `src.neural.dataset` does not exist.

- [ ] **Step 3: Implement dataset metadata helpers**

Create `src/neural/dataset.py`:

```python
from dataclasses import dataclass


@dataclass
class PairExample:
    query_song_id: int
    candidate_song_id: int
    query_file_path: str
    candidate_file_path: str
    query_start_seconds: float
    candidate_start_seconds: float
    query_valid_seconds: float
    padding_ratio: float
    pair_type: str
    label: int


def make_same_time_positive(
    song: dict,
    start_seconds: float,
    query_valid_seconds: float,
    window_seconds: float = 5.0,
) -> PairExample:
    return PairExample(
        query_song_id=int(song["song_id"]),
        candidate_song_id=int(song["song_id"]),
        query_file_path=str(song["file_path"]),
        candidate_file_path=str(song["file_path"]),
        query_start_seconds=start_seconds,
        candidate_start_seconds=start_seconds,
        query_valid_seconds=query_valid_seconds,
        padding_ratio=padding_ratio(query_valid_seconds, window_seconds),
        pair_type="positive_same_time",
        label=1,
    )


def make_jittered_positive(
    song: dict,
    query_start_seconds: float,
    jitter_seconds: float,
    query_valid_seconds: float,
    window_seconds: float = 5.0,
) -> PairExample:
    return PairExample(
        query_song_id=int(song["song_id"]),
        candidate_song_id=int(song["song_id"]),
        query_file_path=str(song["file_path"]),
        candidate_file_path=str(song["file_path"]),
        query_start_seconds=query_start_seconds,
        candidate_start_seconds=max(0.0, query_start_seconds + jitter_seconds),
        query_valid_seconds=query_valid_seconds,
        padding_ratio=padding_ratio(query_valid_seconds, window_seconds),
        pair_type="positive_jittered",
        label=1,
    )


def make_random_negative(
    query_song: dict,
    candidate_song: dict,
    query_start_seconds: float,
    candidate_start_seconds: float,
    query_valid_seconds: float,
    window_seconds: float = 5.0,
) -> PairExample:
    if int(query_song["song_id"]) == int(candidate_song["song_id"]):
        raise ValueError("random negative requires different songs")
    return PairExample(
        query_song_id=int(query_song["song_id"]),
        candidate_song_id=int(candidate_song["song_id"]),
        query_file_path=str(query_song["file_path"]),
        candidate_file_path=str(candidate_song["file_path"]),
        query_start_seconds=query_start_seconds,
        candidate_start_seconds=candidate_start_seconds,
        query_valid_seconds=query_valid_seconds,
        padding_ratio=padding_ratio(query_valid_seconds, window_seconds),
        pair_type="negative_random",
        label=0,
    )


def make_hard_negative(
    query_song: dict,
    candidate_song: dict,
    query_start_seconds: float,
    candidate_offset_seconds: float,
    query_valid_seconds: float,
    window_seconds: float = 5.0,
) -> PairExample:
    if int(query_song["song_id"]) == int(candidate_song["song_id"]):
        raise ValueError("hard negative requires different songs")
    return PairExample(
        query_song_id=int(query_song["song_id"]),
        candidate_song_id=int(candidate_song["song_id"]),
        query_file_path=str(query_song["file_path"]),
        candidate_file_path=str(candidate_song["file_path"]),
        query_start_seconds=query_start_seconds,
        candidate_start_seconds=candidate_offset_seconds,
        query_valid_seconds=query_valid_seconds,
        padding_ratio=padding_ratio(query_valid_seconds, window_seconds),
        pair_type="negative_hard",
        label=0,
    )


def padding_ratio(valid_seconds: float, window_seconds: float) -> float:
    return max(0.0, window_seconds - valid_seconds) / window_seconds


def duration_bucket(valid_seconds: float) -> str:
    if valid_seconds >= 5.0:
        return "5.0s"
    if valid_seconds >= 4.0:
        return "4-5s"
    if valid_seconds >= 3.0:
        return "3-4s"
    return "2-3s"
```

- [ ] **Step 4: Run dataset tests**

Run:

```powershell
python -m unittest tests.test_neural_dataset
```

Expected: PASS.

- [ ] **Step 5: Commit Task 8**

```powershell
git add src\neural\dataset.py tests\test_neural_dataset.py
git commit -m "Add neural pair dataset metadata helpers"
```

---

### Task 9: Add Evaluation Metrics

**Files:**
- Create: `src/neural/evaluation.py`
- Test: `tests/test_neural_evaluation.py`

- [ ] **Step 1: Write failing evaluation tests**

Create `tests/test_neural_evaluation.py`:

```python
import unittest

from src.neural.evaluation import confusion_at_threshold, grouped_confusion


class NeuralEvaluationTests(unittest.TestCase):
    def test_confusion_at_threshold_counts_binary_outcomes(self):
        rows = [
            {"label": 1, "probability": 0.9, "pair_type": "positive_same_time"},
            {"label": 1, "probability": 0.2, "pair_type": "positive_jittered"},
            {"label": 0, "probability": 0.8, "pair_type": "negative_hard"},
            {"label": 0, "probability": 0.1, "pair_type": "negative_random"},
        ]

        report = confusion_at_threshold(rows, threshold=0.7)

        self.assertEqual(1, report["TP"])
        self.assertEqual(1, report["FN"])
        self.assertEqual(1, report["FP"])
        self.assertEqual(1, report["TN"])
        self.assertEqual(0.5, report["precision"])
        self.assertEqual(0.5, report["recall"])

    def test_grouped_confusion_reports_each_group(self):
        rows = [
            {"label": 1, "probability": 0.9, "pair_type": "positive_same_time"},
            {"label": 0, "probability": 0.1, "pair_type": "negative_random"},
        ]

        grouped = grouped_confusion(rows, threshold=0.7, group_key="pair_type")

        self.assertEqual({"negative_random", "positive_same_time"}, set(grouped.keys()))
        self.assertEqual(1, grouped["positive_same_time"]["TP"])
        self.assertEqual(1, grouped["negative_random"]["TN"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run failing evaluation tests**

Run:

```powershell
python -m unittest tests.test_neural_evaluation
```

Expected: FAIL because `src.neural.evaluation` does not exist.

- [ ] **Step 3: Implement evaluation helpers**

Create `src/neural/evaluation.py`:

```python
from collections import defaultdict
from typing import Iterable


def confusion_at_threshold(rows: Iterable[dict], threshold: float) -> dict:
    tp = fp = fn = tn = 0
    for row in rows:
        label = int(row["label"])
        predicted_same = float(row["probability"]) >= threshold
        if predicted_same and label == 1:
            tp += 1
        elif predicted_same and label == 0:
            fp += 1
        elif not predicted_same and label == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def grouped_confusion(rows: Iterable[dict], threshold: float, group_key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return {
        group: confusion_at_threshold(group_rows, threshold)
        for group, group_rows in groups.items()
    }
```

- [ ] **Step 4: Run evaluation tests**

Run:

```powershell
python -m unittest tests.test_neural_evaluation
```

Expected: PASS.

- [ ] **Step 5: Commit Task 9**

```powershell
git add src\neural\evaluation.py tests\test_neural_evaluation.py
git commit -m "Add neural evaluation metrics"
```

---

### Task 10: Final Verification

**Files:**
- Verify all modified source and test files.

- [ ] **Step 1: Run focused neural and recognition tests**

Run:

```powershell
python -m unittest tests.test_top_candidates_trace tests.test_neural_audio_windows tests.test_neural_features tests.test_neural_model tests.test_neural_validator tests.test_neural_dataset tests.test_neural_evaluation tests.test_neural_shadow_service tests.test_search_offset_fallback
```

Expected: PASS for all listed tests.

- [ ] **Step 2: Run the full test suite**

Run:

```powershell
python -m unittest
```

Expected: PASS for the full suite.

- [ ] **Step 3: Compile changed Python modules**

Run:

```powershell
python -m compileall src tests
```

Expected: command exits with code 0 and no syntax errors.

- [ ] **Step 4: Review git diff**

Run:

```powershell
git diff --stat
git diff -- src\recognition\search_trace.py src\recognition\query_pipeline.py src\recognition\service.py src\neural tests config.py
```

Expected: diff only contains neural shadow validator work and tests.

- [ ] **Step 5: Commit final verification note if fixes were needed**

If Step 1-4 required fixes, commit those fixes:

```powershell
git add config.py src\recognition src\neural tests
git commit -m "Stabilize neural shadow validator implementation"
```

If no fixes were needed, do not create an empty commit.

---

## Self-Review

- Spec coverage:
  - Top-N candidates: Task 2.
  - Trace additions: Task 1.
  - Query/candidate 5-second zero-padding policy: Task 3.
  - `[query, candidate, abs_diff]` features: Task 4.
  - Pair classifier model: Task 5.
  - Shadow validator: Task 6.
  - Service integration preserving fingerprint result: Task 7.
  - Dataset pair metadata, duration buckets, hard/random pair types: Task 8.
  - Confusion metrics and grouped reporting foundation: Task 9.
  - Verification: Task 10.
- Scope intentionally excluded from this plan:
  - Full training CLI.
  - Real hard-negative persistence format.
  - Gray-zone decision behavior that changes search results.
  - Audio caching or precomputed features.
- Type consistency:
  - `CandidateTrace` and `NeuralCandidateTrace` are defined in Task 1 and reused consistently.
  - `NeuralValidationResult` is defined in Task 6 and consumed in Task 7.
  - Config names introduced in Task 2 are reused in Task 6.
