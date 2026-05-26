import librosa
import numpy as np
from typing import Any
from pathlib import Path

import config as cfg

def stft(audio:Any, n_fft:int=cfg.N_FFT, hop_length:int=cfg.HOP_LENGTH, window:str=cfg.WINDOW, normalize:bool = True) -> np.ndarray:
    if audio is None:
        raise ValueError("Audio is None")

    if len(audio) == 0:
        raise ValueError("Audio array is empty")

    if len(audio) < n_fft:
        raise ValueError(
            f"Audio too short for STFT: len(audio)={len(audio)}, n_fft={n_fft}"
        )
    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
    X = np.abs(X)
    X = np.log1p(X)
    X -= np.min(X)
    max_value = np.max(X)
    if max_value > 0:
        X /= max_value
    return X

#spectrogram data: [time x freq] -> list of TimeFreqPoint: time x freq
def filter_spectrogram(data: np.ndarray) -> np.ndarray:
    band_edges = [
        (0, 15),
        (15, 30),
        (30, 60),
        (60, 120),
        (120, 240),
        (240, 512),
    ]

    n_times = data.shape[0]
    n_bands = len(band_edges)

    strongest = np.zeros((n_times, n_bands, 2), dtype=float)

    for t in range(n_times):
        spectrum = data[t]

        for b, (low, high) in enumerate(band_edges):
            band = spectrum[low:high]

            if band.size == 0:
                continue

            idx = int(np.argmax(band))
            strongest[t, b, 0] = low + idx
            strongest[t, b, 1] = band[idx]

    thresholds = np.zeros(n_bands, dtype=float)

    for b in range(n_bands):
        # threshold по максимумам в этой полосе, а не по всем bin-ам
        thresholds[b] = np.percentile(strongest[:, b, 1], 75)

    result: list[tuple[int, int]] = []

    for t in range(n_times):
        for b in range(n_bands):
            amp = strongest[t, b, 1]

            if amp >= thresholds[b]:
                freq = int(strongest[t, b, 0])
                result.append((t, freq))

    return np.array(result, dtype=int)


