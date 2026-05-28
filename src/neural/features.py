import librosa
import numpy as np


def log_mel(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio must contain only finite values")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    values = librosa.power_to_db(mel, ref=np.max)
    mean = float(values.mean())
    std = float(values.std())
    return ((values - mean) / (std + 1e-6)).astype(np.float32)


def build_pair_features(
    query_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    query_mel = log_mel(query_audio, sample_rate, n_mels, n_fft, hop_length)
    candidate_mel = log_mel(candidate_audio, sample_rate, n_mels, n_fft, hop_length)
    frame_count = min(query_mel.shape[1], candidate_mel.shape[1])
    if frame_count == 0:
        raise ValueError("log-mel features must have at least one frame")

    query_mel = query_mel[:, :frame_count]
    candidate_mel = candidate_mel[:, :frame_count]
    difference = np.abs(query_mel - candidate_mel)
    return np.stack([query_mel, candidate_mel, difference]).astype(np.float32)
