"""Core helpers for neural pair-classifier training."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

import config as cfg
from src.neural.augmentations import QueryAugmentationConfig
from src.neural.evaluation import confusion_at_threshold
from src.neural.model import PairClassifier
from src.neural.splits import load_song_split
from src.neural.training_data import NeuralPairDataset, make_training_loader


@dataclass
class TrainingConfig:
    sample_rate: int = cfg.SAMPLE_RATE
    n_mels: int = cfg.NEURAL_N_MELS
    n_fft: int = cfg.NEURAL_MEL_N_FFT
    hop_length: int = cfg.NEURAL_MEL_HOP_LENGTH
    input_mode: str = cfg.NEURAL_INPUT_MODE
    batch_size: int = cfg.NEURAL_TRAIN_BATCH_SIZE
    epochs: int = cfg.NEURAL_TRAIN_EPOCHS
    learning_rate: float = cfg.NEURAL_TRAIN_LR
    weight_decay: float = cfg.NEURAL_TRAIN_WEIGHT_DECAY
    mixed_precision: bool = cfg.NEURAL_TRAIN_MIXED_PRECISION
    num_workers: int = cfg.NEURAL_TRAIN_NUM_WORKERS
    split_path: Path = cfg.NEURAL_SPLIT_PATH
    examples_per_epoch: int = getattr(cfg, "NEURAL_TRAIN_EXAMPLES_PER_EPOCH", 4096)
    validation_examples: int = getattr(cfg, "NEURAL_VALIDATION_EXAMPLES", 512)


def evaluate_logits(
    logits,
    labels,
    pair_type: Sequence[str] | None = None,
    duration_bucket: Sequence[str] | None = None,
) -> list[dict]:
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
    labels_tensor = torch.as_tensor(labels).reshape(-1)
    if logits_tensor.numel() != labels_tensor.numel():
        raise ValueError("logits and labels must have the same length")

    probabilities = torch.sigmoid(logits_tensor)
    rows: list[dict] = []
    for index, probability in enumerate(probabilities):
        row = {
            "probability": float(probability.item()),
            "label": int(labels_tensor[index].item()),
        }
        if pair_type is not None:
            row["pair_type"] = str(pair_type[index])
        if duration_bucket is not None:
            row["duration_bucket"] = str(duration_bucket[index])
        rows.append(row)
    return rows


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixed_precision: bool,
) -> float:
    model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    batch_count = 0
    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for features, labels in batches:
        features = features.to(device)
        labels = labels.to(device, dtype=torch.float32).reshape(-1)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(features).reshape(-1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().item())
        batch_count += 1

    return total_loss / batch_count if batch_count else 0.0


def evaluate_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    threshold_values=(0.5, 0.7, 0.85),
) -> dict:
    model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_examples = 0
    logits_parts = []
    labels_parts = []
    pair_types: list[str] = []
    duration_buckets: list[str] = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device, dtype=torch.float32).reshape(-1)
            logits = model(features).reshape(-1)
            loss = criterion(logits, labels)

            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            logits_parts.append(logits.detach().cpu())
            labels_parts.append(labels.detach().cpu())
            pair_types.extend(batch.get("pair_type", []))
            duration_buckets.extend(batch.get("duration_bucket", []))

    if logits_parts:
        rows = evaluate_logits(
            torch.cat(logits_parts),
            torch.cat(labels_parts),
            pair_types,
            duration_buckets,
        )
    else:
        rows = []

    return {
        "loss": total_loss / total_examples if total_examples else 0.0,
        "rows": rows,
        "thresholds": {
            float(threshold): confusion_at_threshold(rows, float(threshold))
            for threshold in threshold_values
        },
    }


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainingConfig,
    best_metric: float | None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "best_metric": best_metric,
        },
        output_path,
    )


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float | None]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint["epoch"]), checkpoint.get("best_metric")


def run_training(args) -> int:
    split_path = Path(args.split)
    try:
        split = load_song_split(split_path)
    except FileNotFoundError:
        print(f"Split file not found: {split_path}")
        print("Run neural-split before neural-train.")
        return 1
    except (KeyError, TypeError, ValueError) as error:
        print(f"Invalid split file: {split_path}")
        print(f"  {error}")
        return 1

    device = _resolve_training_device(args.device)
    if device is None:
        print("CUDA was requested for neural training, but CUDA is not available.")
        print("Use --device cpu or --device auto on this machine.")
        return 1

    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        split_path=split_path,
        num_workers=_arg_or_default(
            args,
            "num_workers",
            getattr(cfg, "NEURAL_TRAIN_NUM_WORKERS", TrainingConfig.num_workers),
        ),
        examples_per_epoch=_arg_or_default(
            args,
            "examples_per_epoch",
            getattr(cfg, "NEURAL_TRAIN_EXAMPLES_PER_EPOCH", TrainingConfig.examples_per_epoch),
        ),
        validation_examples=_arg_or_default(
            args,
            "validation_examples",
            getattr(cfg, "NEURAL_VALIDATION_EXAMPLES", TrainingConfig.validation_examples),
        ),
    )

    print("Neural training configuration:")
    print(f"  Split file: {config.split_path}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Device: {device}")
    print(f"  Mixed precision: {config.mixed_precision and device.type == 'cuda'}")

    try:
        validation_augmentation_config = QueryAugmentationConfig(enabled=False)
        train_dataset = NeuralPairDataset(
            split.train,
            examples_per_epoch=config.examples_per_epoch,
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
        validation_known_dataset = NeuralPairDataset(
            split.train,
            examples_per_epoch=config.validation_examples,
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            query_augmentation_config=validation_augmentation_config,
        )
        validation_heldout_dataset = NeuralPairDataset(
            split.validation_heldout,
            examples_per_epoch=config.validation_examples,
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            query_augmentation_config=validation_augmentation_config,
        )
    except ValueError as error:
        print(f"Cannot create neural training datasets: {error}")
        return 1

    train_loader = make_training_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_known_loader = make_training_loader(
        validation_known_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    validation_heldout_loader = make_training_loader(
        validation_heldout_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = PairClassifier(input_channels=2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_validation_loss: float | None = None
    start_epoch = 1
    checkpoint_path = Path(cfg.NEURAL_MODEL_PATH)
    if checkpoint_path.exists() and not getattr(args, "fresh", False):
        try:
            checkpoint_epoch, best_validation_loss = load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
            )
        except (RuntimeError, KeyError, TypeError, ValueError) as error:
            print(f"Cannot resume neural checkpoint: {checkpoint_path}")
            print(f"  {error}")
            print("Use --fresh to start from a new model.")
            return 1

        start_epoch = checkpoint_epoch + 1
        print(f"Resumed checkpoint: {checkpoint_path}")
        print(f"  Starting epoch: {start_epoch}")

    try:
        final_epoch = start_epoch + config.epochs - 1
        for epoch in range(start_epoch, final_epoch + 1):
            train_loss = train_one_epoch(
                model,
                _feature_label_batches(train_loader),
                optimizer,
                device,
                config.mixed_precision,
            )
            validation_known = evaluate_loader(model, validation_known_loader, device)
            validation_heldout = evaluate_loader(model, validation_heldout_loader, device)
            validation_loss = float(validation_heldout["loss"])

            print(
                f"Epoch {epoch}/{final_epoch}: "
                f"train_loss={train_loss:.4f} "
                f"validation_known_loss={validation_known['loss']:.4f} "
                f"validation_heldout_loss={validation_loss:.4f}"
            )

            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(
                    cfg.NEURAL_MODEL_PATH,
                    model,
                    optimizer,
                    epoch=epoch,
                    config=config,
                    best_metric=best_validation_loss,
                )
                print(f"  Saved checkpoint: {cfg.NEURAL_MODEL_PATH}")
    except (RuntimeError, ValueError) as error:
        print(f"Neural training failed: {_format_error(error)}")
        return 1

    return 0


def _feature_label_batches(loader):
    for batch in loader:
        yield batch["features"], batch["labels"]


def _arg_or_default(args, name: str, default):
    value = getattr(args, name, None)
    return default if value is None else value


def _format_error(error: BaseException) -> str:
    try:
        message = str(error)
    except Exception:
        message = repr(error)
    return message or repr(error)


def _resolve_training_device(device_name: str) -> torch.device | None:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return None
    return torch.device(device_name)
