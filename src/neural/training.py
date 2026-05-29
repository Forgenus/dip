"""Neural model training helpers and configuration.

Core training utilities for the pair classifier model, including training loop,
checkpoint management, and evaluation helpers.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

import config as cfg


@dataclass
class TrainingConfig:
    """Training configuration using config defaults."""

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
    train_ratio: float = cfg.NEURAL_SPLIT_TRAIN_RATIO
    validation_ratio: float = cfg.NEURAL_SPLIT_VALIDATION_RATIO
    test_ratio: float = cfg.NEURAL_SPLIT_TEST_RATIO


def evaluate_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    pair_type: Optional[np.ndarray] = None,
    duration_bucket: Optional[np.ndarray] = None,
) -> list[dict[str, Any]]:
    """Evaluate logits and return rows with probability, label, pair type, and duration.

    Args:
        logits: Model logits [batch_size].
        labels: Binary labels [batch_size].
        pair_type: Optional pair type labels [batch_size].
        duration_bucket: Optional duration bucket labels [batch_size].

    Returns:
        List of dicts with keys: probability, label, pair_type, duration_bucket.
    """
    if len(logits) != len(labels):
        raise ValueError("logits and labels must have the same length")

    probabilities = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    rows = []
    for i in range(len(logits)):
        row = {
            "probability": float(probabilities[i]),
            "label": int(labels[i]),
        }
        if pair_type is not None:
            row["pair_type"] = str(pair_type[i])
        if duration_bucket is not None:
            row["duration_bucket"] = str(duration_bucket[i])
        rows.append(row)

    return rows


def train_one_epoch(
    model: nn.Module,
    batches: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer,
    device: torch.device,
    mixed_precision: bool = False,
) -> float:
    """Train the model for one epoch.

    Args:
        model: The pair classifier model.
        batches: Iterator of (features, labels) tuples.
        optimizer: Adam or SGD optimizer.
        device: Training device (cpu or cuda).
        mixed_precision: If True and device is cuda, use AMP.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    num_batches = 0

    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for features, labels in batches:
        features = features.to(device)
        labels = labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    config: TrainingConfig,
    best_metric: Optional[float] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path: Path to save the checkpoint.
        model: The model to save.
        optimizer: The optimizer state to save.
        epoch: Current epoch number.
        config: Training configuration.
        best_metric: Optional best metric value (e.g., validation loss).
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "best_metric": best_metric,
    }

    torch.save(checkpoint, output_path)


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
) -> Tuple[int, Optional[float]]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: The model to load state into.
        optimizer: Optional optimizer to load state into.

    Returns:
        Tuple of (epoch, best_metric).
    """
    checkpoint = torch.load(path, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", None)

    return epoch, best_metric
