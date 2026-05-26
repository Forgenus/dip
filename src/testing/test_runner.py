# test_runner.py
import librosa
import soundfile as sf
from pathlib import Path
from src.recognition.service import MusicRecognitionService
from src.testing.reporting import format_search_trace
from src.testing.snippets import _get_snippet_from_file
import config as cfg
import numpy as np
import time
def _get_snippet_from_file(
    file_path: Path,
    snippet_duration: float,
    rng,
) -> np.ndarray:
    sr = cfg.SAMPLE_RATE

    full_audio, _ = librosa.load(file_path, sr=sr, mono=True)

    if len(full_audio) == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    snippet_samples = int(snippet_duration * sr)

    if snippet_samples <= 0:
        raise ValueError(f"Invalid snippet_duration: {snippet_duration}")

    if len(full_audio) <= snippet_samples:
        return full_audio

    max_start_sample = len(full_audio) - snippet_samples
    start_sample = rng.integers(0, max_start_sample + 1)
    end_sample = start_sample + snippet_samples

    return full_audio[start_sample:end_sample]

def _save_to_file(audio, output_file: Path, sr: int = cfg.SAMPLE_RATE):
    sf.write(output_file, audio, sr, subtype='FLOAT')


class TestRunner:
    rng: np.random.Generator = np.random.default_rng(cfg.RNG_SEED)
    def __init__(self, service: MusicRecognitionService):
        self.service = service

    def run(self, args):
        if args.reset_db:
            self.service.clear_all()
            self.service.add_songs_from_folder(cfg.SONGS_DIR, max_amount=args.max_files)

        if not args.keep_rng:
            TestRunner.rng = np.random.default_rng(cfg.RNG_SEED)

        correct = 0
        count = args.test_count
        total_time = 0.0

        for _ in range(count):
            song = self.service.get_random_song(rng=TestRunner.rng)
            path = song['file_path']
            song_duration = float(song['duration'])

            snippet = _get_snippet_from_file(
                file_path=path,
                snippet_duration=args.snippet_duration,
                rng=TestRunner.rng
            )

            start = time.perf_counter()
            found_id, time_offset = self.service.search_song(
                snippet,
                _debug_correct_id=song['song_id'],
                file_path=path
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed

            found = self.service.get_song_by_id(found_id)
            expected = song['title']
            result = found['title'] if found else 'nothing'
            match = found_id == song['song_id']

            if match:
                correct += 1
            else:
                print(f"{'Y' if match else 'N'} expected={expected} {song['song_id']} | found={result} {found_id} ")

                trace = self.service.last_search_trace

                if trace is not None:
                    print(format_search_trace(trace))

        avg_time = total_time / count if count > 0 else 0.0

        print(f"\nAccuracy: {correct}/{count} ({correct/count*100:.1f}%)")
        print(f"Average query time: {avg_time:.4f} sec")