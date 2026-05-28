# Neural Model And Training Design

## Purpose

Refine the neural validator model before training and add a dedicated training module.

This design updates the first shadow-validator MVP with:

- Symmetric pair features so swapping query/candidate cannot change the input.
- A richer log-mel representation that is still small enough for local training.
- A stronger CNN sized for an RTX 4070 without wasting memory.
- A single torchaudio-based feature backend for training and runtime to avoid train/inference feature drift.
- A training module that can train, validate, save checkpoints, and produce threshold reports.

The neural validator still remains a shadow/gray-zone component. It does not replace fingerprint search.

## Key Decisions

### Feature Backend

Use one neural feature backend everywhere:

```text
training: torchaudio on GPU in batches
runtime: same torchaudio extractor on CPU or GPU
```

Do not train with torchaudio and infer with librosa. Mixing backends risks train/inference skew because implementations can differ in padding, mel filter scaling, dB conversion, window handling, and floating point behavior.

Do not reuse the fingerprint STFT for the first neural training version. Fingerprint and neural validation optimize for different goals:

- Fingerprint STFT/hop supports stable hashes and offset bins.
- Neural log-mel supports pair similarity validation with more temporal detail.

Keeping a separate neural extractor allows changing neural parameters without rebuilding the fingerprint database.

### Neural Feature Parameters

Keep fingerprint parameters unchanged:

```text
HOP_LENGTH = 768
N_FFT = 1024
```

Use separate neural mel parameters:

```text
NEURAL_N_MELS = 80
NEURAL_MEL_HOP_LENGTH = 384
NEURAL_MEL_N_FFT = 1024
NEURAL_WINDOW_SECONDS = 5.0
SAMPLE_RATE = 11025
```

Approximate size:

```text
samples = 11025 * 5 = 55125
frames ~= 55125 / 384 ~= 144
single logmel ~= [80, 144]
pair input ~= [2, 80, 144]
```

This is a middle ground:

- More temporal detail than the previous `64 x ~72` representation.
- Not as large as `96 x ~216` or `128 x ~216`.
- Practical for 700 songs and an RTX 4070.

## Symmetric Pair Input

Replace the previous non-symmetric input:

```text
[A, B, abs(A - B)]
```

with:

```text
mean     = (A + B) / 2
abs_diff = abs(A - B)
input    = [mean, abs_diff]
```

Properties:

- `features(A, B) == features(B, A)`.
- Model output is invariant to swapping query/candidate by construction.
- `abs_diff` captures local time-frequency mismatch.
- `mean` preserves the shared musical context, helping the model distinguish informative matches from low-information matches such as silence or simple loops.

Do not include `product = A * B` in the first training version. It may act like a local correlation feature, but its interpretation is weaker after normalized log-mel values because negative-negative products become positive. Start with the simpler, explainable two-channel representation.

## Model Architecture

The model returns raw logits, not sigmoid probabilities.

Input:

```text
[batch, 2, 80, 144]
```

Architecture:

```text
ConvBlock 2 -> 32
ConvBlock 32 -> 32
MaxPool2d(2)
shape: [B, 32, 40, 72]

ConvBlock 32 -> 64
ConvBlock 64 -> 64
MaxPool2d(2)
shape: [B, 64, 20, 36]

ConvBlock 64 -> 128
ConvBlock 128 -> 128
MaxPool2d(2)
shape: [B, 128, 10, 18]

ConvBlock 128 -> 192
ConvBlock 192 -> 192
MaxPool2d(2)
shape: [B, 192, 5, 9]

ConvBlock 192 -> 256
ConvBlock 256 -> 256
shape: [B, 256, 5, 9]

GlobalAvgPool + GlobalMaxPool
shape: [B, 512]

Linear 512 -> 256
Dropout 0.3
Linear 256 -> 64
Dropout 0.2
Linear 64 -> 1
output logits: [B]
```

ConvBlock:

```text
Conv2d
BatchNorm2d or GroupNorm
SiLU or ReLU
```

Recommended default:

- `BatchNorm2d` for normal batch sizes.
- `SiLU` if training speed is acceptable, otherwise `ReLU`.

## Why Logits Instead Of Sigmoid In The Model

Do not put `Sigmoid` inside the model for training.

Preferred training path:

```text
model(input) -> logits
BCEWithLogitsLoss(logits, labels)
```

Probability path:

```text
probability = sigmoid(logits)
```

Why this is better:

- `BCEWithLogitsLoss` combines sigmoid and binary cross entropy in a numerically stable formula.
- Separate `Sigmoid` can saturate for large positive or negative logits.
- Saturated probabilities near `0` or `1` can produce unstable logs and weaker gradients.
- Inference still gets probabilities by applying sigmoid after the model.

This means the validator should apply sigmoid during inference/evaluation, not the model itself during training.

## GPU Memory Estimate

RTX 4070 usually has 12 GB VRAM.

For batch size 128:

```text
input [128, 2, 80, 144]
128 * 2 * 80 * 144 = 2,949,120 float32
input memory ~= 11.8 MB
```

Largest activation before pooling:

```text
[128, 32, 80, 144]
128 * 32 * 80 * 144 ~= 47.2M floats ~= 189 MB
```

Approximate activations:

```text
Block1: ~189 MB
Block2: ~94 MB
Block3: ~47 MB
Block4: ~18 MB
Block5: ~6 MB
```

Training stores activations, gradients, optimizer state, CUDA workspace, and temporary tensors. A practical estimate for batch 128 is still comfortably below 12 GB, especially with mixed precision.

Recommended defaults:

```text
batch_size = 128
mixed_precision = enabled
optimizer = AdamW
```

Then try:

```text
batch_size = 256
```

if data loading and feature extraction keep up.

Expected bottleneck is more likely:

- audio loading from disk;
- feature extraction;
- DataLoader throughput;
- CPU preprocessing if not batched on GPU.

## Training Dataset

700 songs should be enough for a diploma-quality validation effect if pair generation is done well.

Reasoning:

```text
700 songs * ~3 minutes/song
~36 non-overlapping 5s windows per song
~25k base windows
many more generated pairs through augmentation and negative sampling
```

The task is binary pair validation, not 700-class classification.

Training split:

```text
train songs: ~560
heldout songs: ~100-140
validation_known: train songs, different windows/augmentations
validation_heldout: songs excluded from training
```

Per epoch target mix:

```text
50% positive pairs
50% negative pairs
```

Positive pairs:

```text
70% same-time positives
30% jittered positives within +/- 0.5s
```

Negative pairs initially:

```text
80% random negatives
20% hard negatives
```

Later, after enough fingerprint failures are collected:

```text
50% random negatives
50% hard negatives
```

Query duration augmentation:

```text
70% full 5.0s query
20% 3.0-5.0s query + zero padding
10% 2.0-3.0s query + zero padding
```

Query augmentations:

- gain changes;
- additive noise at varied SNR;
- light time stretch such as `0.97-1.03`;
- optional low-pass/EQ later if reports show sensitivity.

Candidate audio should usually remain cleaner because runtime candidates are read from stored song files.

## Training Module

Add a dedicated training module under `src/neural/training.py` or a small package:

```text
src/neural/training.py
src/neural/training_config.py
src/neural/checkpointing.py
```

Keep the first implementation simple. A single `training.py` is acceptable if it remains readable.

Responsibilities:

1. Build train/validation datasets from song metadata.
2. Generate pair examples on demand.
3. Load audio windows.
4. Apply query augmentations.
5. Build symmetric torchaudio features.
6. Train model with `BCEWithLogitsLoss`.
7. Evaluate thresholds on known and heldout validation.
8. Save checkpoints.
9. Resume from checkpoint.

Training config should include:

```text
device = cuda if available else cpu
batch_size = 128
epochs = 20-40
learning_rate = 1e-3
weight_decay = 1e-4
num_workers = 4
mixed_precision = true
positive_ratio = 0.5
hard_negative_ratio = 0.2
model_path = data/models/neural_pair_classifier.pt
```

Use `AdamW`.

Use scheduler only if needed:

```text
ReduceLROnPlateau on validation loss
```

Early stopping:

```text
patience = 5 epochs
monitor = validation_known hard-negative false positive rate, then validation loss
```

## Evaluation During Training

Report thresholds:

```text
0.50
0.70
0.85
```

Report slices:

- validation set: known / heldout;
- pair type;
- query duration bucket;
- hard negatives;
- jittered positives.

Important metrics:

```text
TP
FP
FN
TN
precision
recall
false_positive_rate
false_negative_rate
```

For future gray-zone use, pay special attention to:

- hard-negative false positive rate;
- weak-positive false negative rate;
- short-query performance.

## A/B Swap Invariance Tests

Feature-level test:

```text
features(A, B) == features(B, A)
```

Model-level test:

```text
model(features(A, B)) == model(features(B, A))
```

Expected tolerance:

```text
max_abs_logit_delta < 1e-6 on CPU for deterministic eval mode
```

Report-level diagnostic:

```text
max_abs_swap_probability_delta
mean_abs_swap_probability_delta
```

The expected value should be zero or near-zero because the input itself is symmetric.

## CPU/GPU Consistency

Use the same torchaudio feature extractor for CPU and GPU.

Add a consistency check:

```text
same audio
features_cpu = extractor(audio, device=cpu)
features_gpu = extractor(audio, device=cuda).cpu()
```

Accept small numerical differences:

```text
feature max_abs_diff < 1e-3
probability abs_diff < 1e-3
decision unchanged at selected thresholds
```

Do not require bit-identical CPU/GPU output.

## Runtime Behavior

Runtime validator should use the same feature extractor as training.

Runtime options:

```text
device = cpu
device = cuda
device = auto
```

Default for shadow mode can remain CPU if latency is acceptable. For training, default to CUDA when available.

The runtime validator should:

1. Crop/pad query and candidate audio.
2. Build symmetric torchaudio features.
3. Run the model.
4. Apply sigmoid to logits.
5. Store `same_probability` and `decision` in `SearchTrace`.

## Config Changes

Recommended defaults:

```text
NEURAL_N_MELS = 80
NEURAL_MEL_HOP_LENGTH = 384
NEURAL_MEL_N_FFT = 1024
NEURAL_INPUT_MODE = symmetric_mean_absdiff
NEURAL_TRAIN_BATCH_SIZE = 128
NEURAL_TRAIN_EPOCHS = 30
NEURAL_TRAIN_LR = 1e-3
NEURAL_TRAIN_WEIGHT_DECAY = 1e-4
NEURAL_TRAIN_MIXED_PRECISION = true
NEURAL_TRAIN_NUM_WORKERS = 4
```

Keep fingerprint config unchanged.

## Risks

1. 700 songs may not produce production-grade generalization.
   It should still be enough for a diploma-quality shadow validator if hard negatives and validation reports are used.

2. Torchaudio CPU/GPU outputs may differ slightly.
   Use tolerance-based consistency checks, not exact equality.

3. Training can bottleneck on audio loading.
   Use DataLoader workers and consider feature caching only if profiling shows a bottleneck.

4. Hard negatives may be scarce early.
   Start with random negatives and gradually add hard negatives from failure logs.

5. Short query padding can create shortcuts.
   Apply duration augmentation to both positive and negative examples and report by duration bucket.

## Non-Goals

- Replacing fingerprint search.
- Reusing fingerprint STFT in the first training implementation.
- Building a vector database.
- Requiring GPU at runtime.
- Guaranteeing bit-identical CPU/GPU features.
- Training on a larger external corpus before proving the 700-song MVP.

