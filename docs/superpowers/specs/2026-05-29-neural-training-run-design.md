# Neural Training Run Design

## Purpose

Make `python main.py neural-train` perform a real training run for the neural pair classifier.

The current project already has the model, symmetric torchaudio features, song splits, pair sampling helpers, checkpoint helpers, and a CLI skeleton. The missing part is the actual training data pipeline and epoch loop that connect those pieces.

This design focuses only on training the local neural validator model. It does not replace fingerprint recognition and does not redesign the existing recognition pipeline.

## Current State

Already available:

- `src/neural/splits.py`: load song-level train/validation/test splits.
- `src/neural/pairs.py`: choose query duration and pair kind.
- `src/neural/dataset.py`: `PairExample` metadata helpers.
- `src/neural/audio_windows.py`: crop/pad query and candidate audio windows.
- `src/neural/features.py`: `TorchMelPairFeatureExtractor`, symmetric `[mean, abs_diff]` features.
- `src/neural/model.py`: `PairClassifier(input_channels=2)` returning logits.
- `src/neural/training.py`: config, `train_one_epoch`, `evaluate_logits`, checkpoint helpers, CLI skeleton.
- `src/neural/evaluation.py`: confusion metrics.
- `src/cli/parser.py` and `src/cli/commands.py`: `neural-train` command exists.

Missing:

- A Dataset/DataLoader that produces real pair examples from split songs.
- Audio loading and window extraction for each pair.
- Batch feature extraction for training.
- Model/optimizer creation inside `run_training`.
- Validation reports and checkpoint saving during training.

## Target Command

Primary command:

```powershell
python main.py neural-train --split data/neural/splits/song_split.json --epochs 30 --batch-size 128 --device auto
```

Expected behavior:

1. Load the split JSON.
2. Build train and validation loaders.
3. Train `PairClassifier(input_channels=2)` with `BCEWithLogitsLoss`.
4. Evaluate after each epoch on:
   - `validation_known`: train songs, fresh validation samples.
   - `validation_heldout`: heldout validation songs.
5. Print concise epoch metrics.
6. Save the best checkpoint to `cfg.NEURAL_MODEL_PATH`.

## Data Model

Add a training dataset class that samples pair examples on demand:

```text
NeuralPairDataset
  split_items: list[SongSplitItem]
  all_items or candidate_items: list[SongSplitItem]
  examples_per_epoch: int
  mode: train | validation_known | validation_heldout
  rng seed
```

Each dataset item returns a small batch-ready object:

```text
features: Tensor[2, n_mels, frames]
label: float
pair_type: str
duration_bucket: str
```

The dataset should not precompute all possible pairs. It should generate pairs lazily so training can produce many combinations from 700 songs.

## Pair Sampling

Use the existing pair kind names:

- `positive_same_time`
- `positive_jittered`
- `negative_random`
- `negative_hard`

Initial training mix:

```text
positive_ratio = 0.5
hard_negative_ratio = 0.2
```

Positive pairs:

- `positive_same_time`: same song, same start.
- `positive_jittered`: same song, candidate start shifted by up to `+/- 0.5s`.

Negative pairs:

- `negative_random`: different songs, random starts.
- `negative_hard`: initially same as random negative unless a future hard-negative source is provided.

This keeps the first real trainer useful without depending on failure logs that may not exist yet.

## Audio Loading And Windows

For each pair:

1. Load query song audio with `src.processing.preprocess.load_audio`.
2. Load candidate song audio with the same sample rate.
3. Choose `query_valid_seconds` using `choose_query_valid_seconds`.
4. Crop query audio from `query_start_seconds` for `query_valid_seconds`, then pad to `NEURAL_WINDOW_SECONDS`.
5. Crop candidate audio from `candidate_start_seconds` for `NEURAL_WINDOW_SECONDS`.
6. Build symmetric features with `TorchMelPairFeatureExtractor`.

For training speed, add a small per-worker in-memory audio cache keyed by file path. This avoids reloading the same song repeatedly inside a worker.

## Feature Extraction Location

Use torchaudio features in the dataset/collate path first.

Reason: it is simpler and matches the runtime validator. The initial implementation can run on CPU or CUDA. If feature extraction becomes the bottleneck, a separate optimization task can move raw windows through the DataLoader and run feature extraction on GPU in larger batches.

The model still receives:

```text
[batch, 2, NEURAL_N_MELS, frames]
```

## Training Loop

`run_training(args)` should:

1. Build `TrainingConfig` from CLI/config defaults.
2. Resolve device:
   - `auto`: CUDA if available, else CPU.
   - `cpu`: CPU.
   - `cuda`: require CUDA, fail clearly if unavailable.
3. Load split with `load_song_split`.
4. Create:
   - `PairClassifier(input_channels=2)`
   - `AdamW`
   - train DataLoader
   - validation loaders
5. For each epoch:
   - call `train_one_epoch`
   - evaluate validation loaders
   - print metrics
   - save best checkpoint

Loss:

```text
BCEWithLogitsLoss(logits, labels)
```

Probabilities are only for evaluation/reporting:

```text
probability = sigmoid(logit)
```

## Validation Metrics

Evaluate thresholds:

```text
0.50
0.70
0.85
```

Print at least:

- validation set name
- threshold
- TP, FP, FN, TN
- precision
- recall
- false_positive_rate
- false_negative_rate

Also group by:

- `pair_type`
- `duration_bucket`

The main checkpoint selection metric should be validation loss. For diploma reporting, hard-negative false positive rate is important, but it should not block the first trainer.

## Checkpoints

Save best checkpoint to:

```text
cfg.NEURAL_MODEL_PATH
```

Checkpoint must include:

- model state
- optimizer state
- epoch
- training config
- best validation metric

The existing `save_checkpoint` / `load_checkpoint` helpers should remain the checkpoint API.

## CLI Scope

Keep the CLI small:

```text
--split
--epochs
--batch-size
--device
```

Do not add many tuning flags yet. Defaults should come from `config.py`.

Optional but useful:

```text
--examples-per-epoch
--validation-examples
```

If omitted, use config defaults added for these counts.

## Error Handling

Fail clearly when:

- split file does not exist;
- split contains no train songs;
- validation split is empty;
- CUDA is requested but unavailable;
- no valid audio window can be produced for a song;
- audio file path in split no longer exists.

Skip individual invalid pair samples only when a replacement can be sampled. If too many replacements fail, raise an error instead of silently training on bad data.

## Tests

Add focused tests for:

1. Dataset returns feature tensor, label, pair type, and duration bucket.
2. Dataset never creates random/hard negatives with the same song.
3. Collation produces `[batch, 2, n_mels, frames]` and labels `[batch]`.
4. `run_training` builds model/optimizer/loaders and saves a checkpoint for a tiny fake split.
5. `--device cuda` fails clearly when CUDA is unavailable.
6. Validation reports include threshold metrics.

Use small synthetic WAV files or patch `pp.load_audio` in unit tests. Do not require real music files for tests.

## Non-Goals

- No full experiment tracking system.
- No external dataset support.
- No feature cache on disk.
- No complex augmentation pipeline beyond query duration and simple jitter.
- No replacement of fingerprint search.
- No hyperparameter search.

## Done Criteria

The task is complete when:

1. `python main.py neural-split --force` can create a split from the current song DB.
2. `python main.py neural-train --epochs 1 --batch-size 4 --device cpu` performs a real training epoch.
3. A checkpoint is written to `cfg.NEURAL_MODEL_PATH`.
4. Validation metrics are printed after the epoch.
5. Unit tests pass.
6. `python -m compileall src tests` passes.
