import librosa
import numpy as np
from typing import  Any
from pathlib import Path
import sys
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
import config as cfg

def stft(audio:Any, n_fft:int=1024, hop_length:int=cfg.HOP_LENGTH, window:str=cfg.WINDOW, normalize:bool = True) -> np.ndarray:
    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
    X = np.abs(X)
    #X = librosa.amplitude_to_db(X, ref=np.max)
    X = np.log1p(X)
    X -= np.min(X)
    X /= np.max(X)

    return X

#spectrogram data: [time x freq] -> list of TimeFreqPoint: time x freq
def filter_spectrogram(
    data: np.ndarray,
    threshold_coef: float = 0.7
) -> np.ndarray:
    """
    data: shape (n_times, 512)
    return: array of (time_idx, freq_idx)
    """

    band_edges = [
        (0, 15),
        (15,30),
        (30, 60),
        (60, 120),
        (120, 240),
        (240, 512),
    ]

    n_times = data.shape[0]
    n_bands = len(band_edges)

    # [time, band] -> (freq_idx, amplitude)
    strongest = np.zeros((n_times, n_bands, 2))
    data = librosa.amplitude_to_db(data, ref=np.max)

    # --- step 1–2: strongest bin per band per time frame ---
    for t in range(n_times):
        spectrum = data[t]
        for b, (low, high) in enumerate(band_edges):
            band = spectrum[low:high]
            idx = np.argmax(band)
            strongest[t, b, 0] = low + idx      # freq_idx
            strongest[t, b, 1] = band[idx]     # amplitude

    # --- global max per band ---
    #global_band_max = strongest[:, :, 1].max(axis=0)

    thresholds = np.zeros(len(band_edges))
    for b, (low,high) in enumerate(band_edges):
        #thresholds[b] = np.percentile(data[:,low:high],97) 
        band = data[:,low:high] 
        thresholds[b] = np.max(band) - 20

    
   # maxMean = global_band_max.mean()
   # threshold = maxMean * 0.5

    # --- filtering ---
    result: list[tuple[int, int]] = []
    for t in range(n_times):
        for b in range(n_bands):
            amp = strongest[t, b, 1]
            if amp >= thresholds[b]:
            #if amp>threshold:
                freq = int(strongest[t, b, 0])
                result.append((t, freq))

    return np.array(result, dtype=int)


