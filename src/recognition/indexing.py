from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import logging
import os
import time

from src.processing import fft_filter as ff
from src.processing import fingerprint as fp
from src.processing import preprocess as pp

logger = logging.getLogger(__name__)


@dataclass
class IndexPayload:
    file_path: Path
    song_id: int
    fingerprints: List[Tuple[int, int]]


def compute_payload(file_path: Path, song_id: int) -> IndexPayload:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.perf_counter()
    pid = os.getpid()
    print(f"[worker {pid}] start song_id={song_id} file={file_path}", flush=True)
    audio = pp.load_audio(file_path)

    fingerprints = build_fingerprints_from_audio(audio, song_id)
    print(f"[worker {pid}] done song_id={song_id} elapsed={time.perf_counter() - start:.2f}s file={file_path}", flush=True)

    return IndexPayload(
        file_path=file_path,
        song_id=song_id,
        fingerprints=fingerprints,
    )


def build_fingerprints_from_audio(audio, song_id: int) -> List[Tuple[int, int]]:
    """Shared indexing/query preprocessing path.

    ff.stft already returns magnitude, log-scaled, min-max normalized data with
    shape [freq, time]. Peak picking expects [time, freq].
    """
    spectrogram = ff.stft(audio)
    points = ff.filter_spectrogram(spectrogram.T)
    return fp.create_fingerprints(points, song_id)
