# Neural Training Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Keep this implementation practical: add only focused tests that prove training works, and avoid unrelated cleanup.

**Goal:** Make `python main.py neural-train` run a real training epoch for the neural pair classifier and save a checkpoint.

**Architecture:** Keep the project as a modular monolith. Add a training dataset/loader path under `src/neural`, then wire `run_training` to load splits, train `PairClassifier`, evaluate validation batches, and save the best checkpoint.

**Tech Stack:** Python, NumPy, PyTorch, torchaudio, existing `src.neural` modules, existing CLI.

---

## Scope

Implement only what is needed to train the model from the existing song split:

- real pair dataset;
- audio window loading;
- symmetric feature extraction;
- train/validation loop;
- checkpoint save;
- concise CLI output.

Do not add experiment tracking, disk feature caches, hyperparameter search, or broad refactors.

---

## Files

- Modify: `config.py`
  - Add `NEURAL_TRAIN_EXAMPLES_PER_EPOCH`
  - Add `NEURAL_VALIDATION_EXAMPLES`
  - Add `NEURAL_POSITIVE_RATIO`
  - Add `NEURAL_HARD_NEGATIVE_RATIO`
  - Add `NEURAL_POSITIVE_JITTER_SECONDS`
- Create: `src/neural/training_data.py`
  - `TrainingBatch`
  - `NeuralPairDataset`
  - `make_training_loader`
- Modify: `src/neural/training.py`
  - Build model/optimizer/loaders in `run_training`
  - Add validation helper
  - Save best checkpoint
- Modify: `src/cli/parser.py`
  - Add `--examples-per-epoch`
  - Add `--validation-examples`
- Test: `tests/test_neural_training_data.py`
- Test: `tests/test_neural_training_run.py`

---

## Task 1: Training Data Pipeline

**Files:**
- Create: `src/neural/training_data.py`
- Test: `tests/test_neural_training_data.py`

Implement a lazy dataset that turns split songs into pair examples and feature tensors.

### Requirements

- `NeuralPairDataset.__len__` returns `examples_per_epoch`.
- `__getitem__` samples one pair using:
  - `sample_pair_kind`
  - `choose_query_valid_seconds`
  - existing `src.neural.dataset` pair metadata helpers.
- Load audio with `src.processing.preprocess.load_audio(path, target_sr=sample_rate)`.
- Resolve relative split paths against `cfg.BASE_DIR`.
- Use an in-memory dict cache per dataset instance: `dict[str, np.ndarray]`.
- Build query/candidate windows with existing audio window helpers.
- Build features with `TorchMelPairFeatureExtractor`.
- Return:

```python
@dataclass
class TrainingBatch:
    features: torch.Tensor
    label: torch.Tensor
    pair_type: str
    duration_bucket: str
```

- Add `make_training_loader(dataset, batch_size, shuffle, num_workers)` with a custom collate function returning:

```python
{
    "features": torch.Tensor,          # [batch, 2, n_mels, frames]
    "labels": torch.Tensor,            # [batch]
    "pair_type": list[str],
    "duration_bucket": list[str],
}
```

### Minimal Tests

Add `tests/test_neural_training_data.py` with patched `pp.load_audio` returning synthetic arrays.

Cover:

1. dataset item has feature shape `[2, n_mels, frames]`, scalar label, pair type, duration bucket;
2. random negative uses different song ids;
3. loader collates two examples into batch tensors.

### Verification

Run:

```powershell
python -m unittest tests.test_neural_training_data
```

Expected: pass.

---

## Task 2: Real Training Loop

**Files:**
- Modify: `src/neural/training.py`
- Test: `tests/test_neural_training_run.py`

Replace the skeleton behavior in `run_training(args)` with a real training run.

### Requirements

`run_training(args)` should:

1. Load split with `load_song_split(args.split)`.
2. Resolve device:
   - `auto`: CUDA if available else CPU.
   - `cpu`: CPU.
   - `cuda`: return error if CUDA unavailable.
3. Create:
   - `PairClassifier(input_channels=2)`
   - `AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)`
   - train dataset from `split.train`
   - validation_known dataset from `split.train`
   - validation_heldout dataset from `split.validation_heldout`
4. Train for `config.epochs`.
5. Evaluate after each epoch.
6. Save checkpoint to `cfg.NEURAL_MODEL_PATH` when validation loss improves.
7. Return `0` on success and `1` on clear configuration/data errors.

Add:

```python
def evaluate_loader(model, loader, device, threshold_values=(0.5, 0.7, 0.85)) -> dict:
    ...
```

The helper should return:

```python
{
    "loss": float,
    "rows": list[dict],
    "thresholds": dict[float, dict],
}
```

Use existing `evaluate_logits` and `confusion_at_threshold`.

### Minimal Tests

Add `tests/test_neural_training_run.py`.

Cover:

1. `run_training` on a tiny split calls through one epoch and writes a checkpoint.
2. `run_training` returns `1` when `--device cuda` is requested but CUDA is unavailable.

Patch audio loading to synthetic audio. Use tiny config values:

- epochs: 1
- batch size: 2
- examples per epoch: 2
- validation examples: 2
- small mel settings if needed for speed.

### Verification

Run:

```powershell
python -m unittest tests.test_neural_training_run tests.test_neural_training
```

Expected: pass.

---

## Task 3: CLI And Config Defaults

**Files:**
- Modify: `config.py`
- Modify: `src/cli/parser.py`
- Test: `tests/test_parser.py`

Add practical defaults:

```python
NEURAL_TRAIN_EXAMPLES_PER_EPOCH = int(os.getenv("NEURAL_TRAIN_EXAMPLES_PER_EPOCH", "4096"))
NEURAL_VALIDATION_EXAMPLES = int(os.getenv("NEURAL_VALIDATION_EXAMPLES", "512"))
NEURAL_POSITIVE_RATIO = float(os.getenv("NEURAL_POSITIVE_RATIO", "0.5"))
NEURAL_HARD_NEGATIVE_RATIO = float(os.getenv("NEURAL_HARD_NEGATIVE_RATIO", "0.2"))
NEURAL_POSITIVE_JITTER_SECONDS = float(os.getenv("NEURAL_POSITIVE_JITTER_SECONDS", "0.5"))
```

Add CLI flags:

```text
--examples-per-epoch
--validation-examples
```

Keep existing flags:

```text
--split
--epochs
--batch-size
--device
```

### Minimal Tests

Update parser tests to confirm the new flags parse correctly.

### Verification

Run:

```powershell
python -m unittest tests.test_parser
```

Expected: pass.

---

## Task 4: Final Smoke Test

**Files:**
- No new production files unless fixing issues found by the smoke test.

Run focused verification:

```powershell
python -m unittest tests.test_neural_training_data tests.test_neural_training_run tests.test_neural_training tests.test_parser
python -m compileall src tests
```

Then run a tiny command-level smoke test with an existing or test split:

```powershell
python main.py neural-train --epochs 1 --batch-size 2 --device cpu --examples-per-epoch 2 --validation-examples 2
```

Expected:

- command starts a real training loop;
- prints epoch loss and validation metrics;
- writes checkpoint to `cfg.NEURAL_MODEL_PATH`;
- exits with code `0`.

If the real project split does not exist, first run:

```powershell
python main.py neural-split --force
```

Only fix issues that block the training command.

---

## Done Criteria

- `neural-train` no longer prints skeleton-only output.
- One CPU epoch can run from a valid split.
- Checkpoint is written.
- Focused tests pass.
- `python -m compileall src tests` passes.
