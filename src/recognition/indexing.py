from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from src.processing import fft_filter as ff
from src.processing import fingerprint as fp
from src.processing import preprocess as pp


@dataclass
class IndexPayload:
    file_path: Path
    song_id: int
    fingerprints: List[Tuple[int, int]]


def compute_payload(file_path: Path, song_id: int) -> IndexPayload:
    audio = pp.load_audio(file_path)
    spectrogram = ff.stft(audio)
    points = ff.filter_spectrogram(spectrogram.T)
    fingerprints = fp.create_fingerprints(points, song_id)

    return IndexPayload(
        file_path=file_path,
        song_id=song_id,
        fingerprints=fingerprints,
    )
