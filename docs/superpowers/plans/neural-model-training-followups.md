# Neural Model Training - Follow-up Issues

Документ для отслеживания некритичных проблем, найденных при реализации задач 7-10.

## Task 7: Pair Sampling Utilities

*(Нет проблем пока)*

## Task 8: Training Module Core

*(Будут добавлены при выполнении)*

## Task 9: Neural Train CLI Skeleton

*(Будут добавлены при выполнении)*

## Task 10: Final Review
## Final Review - Spec Compliance ✅

All requirements from the plan are implemented:

### Task 7: Pair Sampling Utilities ✅
- `choose_query_valid_seconds(rng, window_seconds=5.0)` ✅
	- Distribution: 70% full window, 20% medium (3-5s), 10% short (2-3s)
	- Deterministic with RNG seed
- `sample_pair_kind(rng, positive_ratio, hard_negative_ratio)` ✅
	- Returns stable pair kind strings for metrics logging
	- Pair kinds: positive_same_time, positive_jittered, negative_random, negative_hard

### Task 8: Training Module Core ✅
- `TrainingConfig` dataclass with all config defaults ✅
- `evaluate_logits(logits, labels, pair_type, duration_bucket)` ✅
	- Returns list of dicts with probability, label, pair_type, duration_bucket
	- Uses sigmoid outside model
- `train_one_epoch(model, batches, optimizer, device, mixed_precision)` ✅
	- BCEWithLogitsLoss for logits training
	- CUDA AMP support when requested and device is CUDA
	- Returns average epoch loss
- `save_checkpoint(path, model, optimizer, epoch, config, best_metric)` ✅
	- Includes all required state
	- Creates parent directories
- `load_checkpoint(path, model, optimizer=None)` ✅
	- Loads model, optimizer (optional), returns epoch and best_metric

### Task 9: Neural Train CLI ✅
- Added `neural-train` parser command ✅
- CLI flags:
	- `--split` (default cfg.NEURAL_SPLIT_PATH) ✅
	- `--epochs` (default cfg.NEURAL_TRAIN_EPOCHS) ✅
	- `--batch-size` (default cfg.NEURAL_TRAIN_BATCH_SIZE) ✅
	- `--device` (choices: auto, cpu, cuda, default: auto) ✅
- `CommandHandler.neural_train(args)` dispatcher ✅
- `run_training(args)` entrypoint helper ✅
	- Validates split file exists
	- Resolves device (auto detects CUDA)
	- Creates TrainingConfig
	- Prints configuration summary
	- Skeleton implementation ready for DataLoader expansion

### Scope Verification ✅
All changes are isolated to neural training module:
- src/neural/pairs.py (new)
- src/neural/training.py (new)
- src/cli/parser.py (modified: added neural-train parser)
- src/cli/commands.py (modified: added neural_train handler)
- No unrelated changes present

### Architecture Cohesion ✅
- Pair sampling uses standard pair kind strings (no enum dependency issues)
- Training uses BCEWithLogitsLoss (external sigmoid in evaluate_logits)
- Features and model already updated to symmetric logits contract (Tasks 4-6)
- CLI follows existing style and conventions
- Ready for DataLoader expansion in follow-up work

*(Будут добавлены при выполнении)*
