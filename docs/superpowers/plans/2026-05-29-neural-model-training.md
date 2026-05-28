# Neural Model Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans`. This plan is review-driven: implementation tasks do not prescribe test writing. Each task must pass a spec compliance review and a code quality review before the next task starts.

**Goal:** Upgrade the neural validator to symmetric torchaudio features, a stronger logits-based CNN, reproducible song splits, and a CLI-driven training module.

**Architecture:** Keep fingerprint recognition unchanged. Add a separate neural feature path based on torchaudio, with song-level split files consumed by training and evaluation. The runtime validator uses the same feature extractor and model contract as training, applying sigmoid outside the model.

**Tech Stack:** Python, NumPy, PyTorch, torchaudio, existing CLI parser/handler, existing `MusicRecognitionService` and song database.

---

## Execution Rules

- Implement one task at a time.
- Do not revert unrelated user or agent changes.
- Commit after each task with the commit message listed in the task.
- After each task, run two reviews:
  - **Spec review:** verify the implementation matches this plan and the design spec.
  - **Code quality review:** verify maintainability, integration safety, edge cases, and API consistency.
- If either review requests changes, fix them and re-review before moving on.
- Tests may be added or run when an implementer/reviewer thinks they are useful, but this plan intentionally does not require per-task test steps.

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

---

### Task 1: Config And Dependency Defaults

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py`

**Implementation:**
- Add `torchaudio` to `requirements.txt`.
- Add these defaults to `config.py`:
  - `NEURAL_N_MELS = 80`
  - `NEURAL_MEL_HOP_LENGTH = 384`
  - `NEURAL_MEL_N_FFT = N_FFT` by default
  - `NEURAL_INPUT_MODE = "symmetric_mean_absdiff"`
  - `NEURAL_TRAIN_BATCH_SIZE = 128`
  - `NEURAL_TRAIN_EPOCHS = 30`
  - `NEURAL_TRAIN_LR = 1e-3`
  - `NEURAL_TRAIN_WEIGHT_DECAY = 1e-4`
  - `NEURAL_TRAIN_MIXED_PRECISION = True`
  - `NEURAL_TRAIN_NUM_WORKERS = 4`
  - `NEURAL_SPLIT_PATH = data/neural/splits/song_split.json`
  - `NEURAL_SPLIT_TRAIN_RATIO = 0.80`
  - `NEURAL_SPLIT_VALIDATION_RATIO = 0.10`
  - `NEURAL_SPLIT_TEST_RATIO = 0.10`
- Use `os.getenv` and existing `get_path` style consistently with the surrounding config.

**Review Criteria:**
- Config names and defaults match the design.
- Existing config behavior remains backward compatible.
- No unrelated settings are changed.

**Commit:** `Add neural training config defaults`

---

### Task 2: Song Split Module

**Files:**
- Create: `src/neural/splits.py`

**Implementation:**
- Add `SongSplitItem` dataclass with `song_id`, `file_path`, `title`, `artist`, `duration`.
- Add `SongSplit` dataclass with `version`, `seed`, `ratios`, `counts`, `train`, `validation_heldout`, `test_heldout`.
- Add `collect_song_items(service_or_db)` that reads songs from the existing song database shape and returns items sorted by `song_id`.
- Add `split_song_items(items, seed, train_ratio, validation_ratio, test_ratio)`:
  - reproducible shuffle via NumPy RNG;
  - song-level split only;
  - default expected count behavior for 80/10/10.
- Add `save_song_split(split, path)` and `load_song_split(path)` using JSON.
- Store portable paths relative to `cfg.BASE_DIR` when possible.

**Review Criteria:**
- Split is deterministic for the same seed.
- No song appears in more than one split.
- JSON is stable and human-readable.
- Function names and dataclasses are suitable for later training code.

**Commit:** `Add neural song split helpers`

---

### Task 3: Neural Split CLI

**Files:**
- Modify: `src/cli/parser.py`
- Modify: `src/cli/commands.py`

**Implementation:**
- Add `neural-split` command.
- CLI flags:
  - `--output`, default `cfg.NEURAL_SPLIT_PATH`
  - `--seed`, default `cfg.RNG_SEED`
  - `--train-ratio`, default `cfg.NEURAL_SPLIT_TRAIN_RATIO`
  - `--validation-ratio`, default `cfg.NEURAL_SPLIT_VALIDATION_RATIO`
  - `--test-ratio`, default `cfg.NEURAL_SPLIT_TEST_RATIO`
  - `--force`
- Add `CommandHandler.neural_split(args)`.
- Refuse to overwrite an existing split unless `--force` is provided.
- Print a short summary with output path and split counts.

**Review Criteria:**
- Parser follows existing CLI conventions.
- Handler uses `src.neural.splits` instead of duplicating split logic.
- Empty song database produces a clear error.
- Existing commands keep working.

**Commit:** `Add neural split CLI`

---

### Task 4: Symmetric Torchaudio Feature Extractor

**Files:**
- Replace: `src/neural/features.py`

**Implementation:**
- Implement `TorchMelPairFeatureExtractor(nn.Module)`.
- Use `torchaudio.transforms.MelSpectrogram` followed by `AmplitudeToDB(stype="power")`.
- Parameters: `sample_rate`, `n_mels`, `n_fft`, `hop_length`.
- Add `build_symmetric_pair_features(left, right)`.
- Output shape must be `[batch, 2, n_mels, frames]`.
- Channel 0: `(left + right) / 2`.
- Channel 1: `abs(left - right)`.
- Cropping/padding behavior must keep pair tensors aligned and avoid order-dependent output.

**Review Criteria:**
- Swapping A/B produces identical features.
- Training and runtime can use the same extractor.
- CPU use is supported for runtime; CUDA tensors should work naturally when inputs/module are on CUDA.
- No librosa dependency remains in the neural feature path.

**Commit:** `Use symmetric torchaudio pair features`

---

### Task 5: Logits Pair Classifier

**Files:**
- Replace: `src/neural/model.py`

**Implementation:**
- Implement a stronger `PairClassifier(input_channels=2)` that returns logits, not probabilities.
- Reject non-positive `input_channels`.
- Use ConvBlocks with `Conv2d`, `BatchNorm2d`, and `SiLU`.
- Suggested channel plan:
  - `2 -> 32 -> 32 -> pool`
  - `32 -> 64 -> 64 -> pool`
  - `64 -> 128 -> 128 -> pool`
  - `128 -> 192 -> 192 -> pool`
  - `192 -> 256 -> 256`
- Pool by concatenating global average and max pooling.
- Suggested MLP head: `512 -> 256 -> 64 -> 1`.
- Do not include `Sigmoid` inside the model.

**Review Criteria:**
- Forward accepts `[batch, 2, 80, 144]`.
- Output shape is `[batch]`.
- Model is symmetric by input contract, not by hidden ordering.
- Logits contract is clear for training and validator.

**Commit:** `Use logits pair classifier architecture`

---

### Task 6: Validator Logits And Symmetric Features

**Files:**
- Modify: `src/neural/validator.py`

**Implementation:**
- Instantiate `TorchMelPairFeatureExtractor` in the validator.
- Load `PairClassifier(input_channels=2)`.
- Convert query/candidate windows to float tensors.
- Run extractor, model, then apply `torch.sigmoid(logits)` outside the model.
- Return rounded probability values as before.

**Review Criteria:**
- Validator uses the same feature/model contract as training.
- Probability threshold behavior remains externally compatible.
- CPU runtime remains supported.
- No old probability-output model assumption remains.

**Commit:** `Use logits and symmetric features in validator`

---

### Task 7: Pair Sampling Utilities

**Files:**
- Create: `src/neural/pairs.py`

**Implementation:**
- Add `choose_query_valid_seconds(rng, window_seconds=5.0)`.
- Distribution:
  - mostly full 5 second snippets;
  - some 3-5 second snippets;
  - fewer 2-3 second snippets.
- Add `sample_pair_kind(rng, positive_ratio, hard_negative_ratio)`.
- Pair kinds:
  - `positive_same_time`
  - `positive_jittered`
  - `negative_random`
  - `negative_hard`

**Review Criteria:**
- Sampling is deterministic for a provided RNG.
- Duration distribution reflects the diploma/runtime target: 5 seconds first, shorter snippets as robustness cases.
- Pair kind names are stable for metrics and future logging.

**Commit:** `Add neural pair sampling helpers`

---

### Task 8: Training Module Core

**Files:**
- Create: `src/neural/training.py`

**Implementation:**
- Add `TrainingConfig` dataclass using config defaults.
- Add `evaluate_logits(logits, labels, pair_type, duration_bucket)` returning rows with probability, label, pair type, and duration bucket.
- Add `train_one_epoch(model, batches, optimizer, device, mixed_precision)`:
  - use `BCEWithLogitsLoss`;
  - support CUDA AMP only when requested and device is CUDA;
  - return average loss.
- Add `save_checkpoint(path, model, optimizer, epoch, config, best_metric)`.
- Add `load_checkpoint(path, model, optimizer=None)`.

**Review Criteria:**
- Sigmoid is used only for reporting/evaluation probabilities, not loss input.
- Checkpoints include model state, optimizer state, epoch, config, and best metric.
- Code works on CPU and can use CUDA when available.

**Commit:** `Add neural training core helpers`

---

### Task 9: Neural Train CLI Skeleton

**Files:**
- Modify: `src/cli/parser.py`
- Modify: `src/cli/commands.py`
- Modify: `src/neural/training.py`

**Implementation:**
- Add `neural-train` command.
- CLI flags:
  - `--split`, default `cfg.NEURAL_SPLIT_PATH`
  - `--epochs`, default `cfg.NEURAL_TRAIN_EPOCHS`
  - `--batch-size`, default `cfg.NEURAL_TRAIN_BATCH_SIZE`
  - `--device`, choices `auto`, `cpu`, `cuda`, default `auto`
- Add `run_training(args)` entrypoint helper.
- Add `CommandHandler.neural_train(args)` that dispatches to `run_training`.
- The initial CLI may be a skeleton that validates/configures training and prints configuration; full real-audio DataLoader can be expanded later.

**Review Criteria:**
- Parser and handler follow existing CLI style.
- Training entrypoint uses `TrainingConfig`.
- The skeleton is honest about what it does and does not train yet.
- No unrelated CLI behavior changes.

**Commit:** `Add neural train CLI entrypoint`

---

### Task 10: Final Review

**Files:**
- Review all changed files.

**Implementation:**
- Run final spec compliance review across the entire branch.
- Run final code quality review across the entire branch.
- Inspect the final diff and ensure it only contains neural model/training/split work.
- Apply any requested review fixes and commit them.

**Review Criteria:**
- Design spec requirements are covered:
  - torchaudio backend;
  - symmetric `[mean, abs_diff]` features;
  - stronger logits model;
  - validator sigmoid outside model;
  - song split module and JSON format;
  - `neural-split` CLI;
  - pair sampling helpers;
  - training checkpoint/eval helpers;
  - `neural-train` CLI entrypoint.
- The implementation is cohesive and ready for a later real-audio DataLoader expansion.

**Commit:** `Stabilize neural model training implementation` only if final review fixes are needed.

---

## Self-Review

- The plan intentionally removes per-task test snippets and failing-test steps.
- Review gates are now the primary required quality mechanism.
- Optional tests are still allowed during implementation, but they are no longer part of the plan contract.
- The scope boundary remains: this creates the training foundation and CLI skeleton; optimized real-audio dataset generation can be a follow-up.
