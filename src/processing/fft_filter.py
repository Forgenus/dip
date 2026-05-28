import numpy as np
import librosa
from typing import Any
from pathlib import Path

import config as cfg

DEFAULT_BAND_EDGES = (
    (0, 15),
    (15, 30),
    (30, 60),
    (60, 120),
    (120, 240),
    (240, 512),
)

DEFAULT_LOG_BAND_COUNT = 32
MAX_ENCODED_FREQ_BIN = 512


def stft(audio:Any, n_fft:int=cfg.N_FFT, hop_length:int=cfg.HOP_LENGTH, window:str=cfg.WINDOW) -> np.ndarray:
    if audio is None:
        raise ValueError("Audio is None")

    if len(audio) == 0:
        raise ValueError("Audio array is empty")

    if len(audio) < n_fft:
        raise ValueError(
            f"Audio too short for STFT: len(audio)={len(audio)}, n_fft={n_fft}"
        )
    X = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=False,
        pad_mode="constant",
    )
    X = np.abs(X)
    X = np.log1p(X)
    return X



def filter_spectrogram(data: np.ndarray, method: str = "stable_peaks") -> np.ndarray:
    if method == "strongest":
        return filter_spectrogram_strongest(data)
    elif method == "log_band_strongest":
        return filter_spectrogram_log_band_strongest(data)
    elif method == "2d_peaks":
        return filter_spectrogram_2d_peaks(data)
    elif method == "stable_peaks":
        return filter_spectrogram_stable_peaks(data)
    else:
        raise ValueError(f"Unknown spectrogram filter method: {method}")





def filter_spectrogram_log_band_strongest(
    data: np.ndarray,
    band_edges: tuple[tuple[int, int], ...] | None = None,
    mean_coefficient: float = 1.2,
) -> np.ndarray:
    n_times, n_freq_bins = data.shape

    if n_times == 0 or n_freq_bins == 0:
        return np.empty((0, 2), dtype=int)

    max_bin = min(n_freq_bins, MAX_ENCODED_FREQ_BIN)
    if band_edges is None:
        band_edges = DEFAULT_BAND_EDGES

    result: list[tuple[int, int]] = []

    for t in range(n_times):
        strongest: list[tuple[int, float]] = []

        for low, high in band_edges:
            low = max(0, low)
            high = min(high, max_bin)

            if low >= high:
                continue

            band = data[t, low:high]
            if band.size == 0:
                continue

            idx = int(np.argmax(band))
            freq = low + idx
            strongest.append((freq, float(band[idx])))

        if not strongest:
            continue

        threshold = float(np.mean([amp for _, amp in strongest])) * mean_coefficient

        for freq, amp in strongest:
            if amp >= threshold:
                result.append((t, freq))

    return np.array(result, dtype=int).reshape(-1, 2)

#spectrogram data: [time x freq] -> list of TimeFreqPoint: time x freq
def filter_spectrogram_strongest(data: np.ndarray, band_edges=DEFAULT_BAND_EDGES) -> np.ndarray:
    n_times = data.shape[0]
    n_freq_bins = data.shape[1]
    n_bands = len(band_edges)

    for low, high in band_edges:
        if low < 0 or high > n_freq_bins or low >= high:
            raise ValueError(
                f"Invalid band edge ({low}, {high}) for {n_freq_bins} frequency bins"
            )

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
                if freq < MAX_ENCODED_FREQ_BIN:
                    result.append((t, freq))

    return np.array(result, dtype=int).reshape(-1, 2)


def filter_spectrogram_2d_peaks(
    data: np.ndarray,
    time_radius: int = 1,
    freq_radius: int = 3,
    percentile: float = 88,
) -> np.ndarray:
    n_times, n_freq_bins = data.shape

    global_threshold = np.percentile(data, percentile)
    result: list[tuple[int, int]] = []

    for t in range(n_times):
        t0 = max(0, t - time_radius)
        t1 = min(n_times, t + time_radius + 1)

        for f in range(min(n_freq_bins, MAX_ENCODED_FREQ_BIN)):
            amp = data[t, f]

            if amp < global_threshold:
                continue

            f0 = max(0, f - freq_radius)
            f1 = min(n_freq_bins, f + freq_radius + 1)

            neighborhood = data[t0:t1, f0:f1]

            if amp >= np.max(neighborhood):
                result.append((t, f))

    return np.array(result, dtype=int).reshape(-1, 2)

def filter_spectrogram_stable_peaks(
    data: np.ndarray,
    band_edges=DEFAULT_BAND_EDGES,
    time_radius: int = 1,
    freq_radius: int = 2,
    percentile: float = 89,
    max_points_per_time: int = 30,
    min_prominence: float = 0.02,
) -> np.ndarray:
    n_times, n_freq_bins = data.shape

    if n_times == 0 or n_freq_bins == 0:
        return np.empty((0, 2), dtype=int)

    points: list[tuple[int, int, float]] = []

    for low, high in band_edges:
        high = min(high, n_freq_bins)

        if low < 0 or low >= high:
            continue

        band_data = data[:, low:high]
        threshold = np.percentile(band_data, percentile)

        for t in range(n_times):
            t0 = max(0, t - time_radius)
            t1 = min(n_times, t + time_radius + 1)

            for f in range(low, high):
                amp = data[t, f]

                if amp < threshold:
                    continue

                f0 = max(low, f - freq_radius)
                f1 = min(high, f + freq_radius + 1)

                neighborhood = data[t0:t1, f0:f1]
                local_max = np.max(neighborhood)

                if amp < local_max:
                    continue

                # peak должен заметно выделяться над локальным фоном
                local_median = np.median(neighborhood)
                if amp - local_median < min_prominence:
                    continue

                points.append((t, f, float(amp)))

    if not points:
        return np.empty((0, 2), dtype=int)

    # ограничить количество пиков на один time-frame
    by_time: dict[int, list[tuple[int, float]]] = {}

    for t, f, amp in points:
        by_time.setdefault(t, []).append((f, amp))

    result: list[tuple[int, int]] = []

    for t, freq_amp in by_time.items():
        freq_amp.sort(key=lambda x: x[1], reverse=True)

        for f, _ in freq_amp[:max_points_per_time]:
            result.append((t, f))

    result.sort(key=lambda x: (x[0], x[1]))
    return np.array(result, dtype=int)
