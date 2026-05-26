"""
Сервисный слой
"""

from pathlib import Path
import sys
from typing import List, Any, Tuple
from ..database import music_database as DB
from ..processing import fft_filter as ff
from ..processing import preprocess as pp
from ..processing import fingerprint as fp
from . import match_filter as mf
import os
import numpy as np
import soundfile as sf
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
import config as cfg
invalidval = 2**32-2
log = print

def compute_payload(file_path: Path, song_id: int) -> dict[str, Any]:
    audio = pp.load_audio(file_path)
    spectrogram = ff.stft(audio)
    points = ff.filter_spectrogram(np.abs(spectrogram).T)
    fingerprints = fp.create_fingerprints(points, song_id)

    return {
        "file_path": file_path,
        "fingerprints": fingerprints,
        "song_id": song_id,
    }
class MusicRecognitionService:
    """
    Сервис распознавания музыки.
    Координирует работу processing и database.
    main.py вызывает ТОЛЬКО этот класс.
    """
    def __init__(self, fp_db_name:str="fingerprints", songs_db_name:str="songs", db_path:Path= cfg.DATABASE_DIR)  -> None:
        # Инициализируем все компоненты
        self.db = DB.MusicDatabase(db_path, fp_db_name, songs_db_name)
        self.metadata = self._load_metadata()

        try:
            self.db.load_all()
        except FileNotFoundError:
            log("No existing database found, starting fresh")

    def _load_metadata(self) -> dict:
        if not cfg.METADATA_JSON_PATH.exists():
            return {}

        try:
            with open(cfg.METADATA_JSON_PATH, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return {}
    @staticmethod
    def get_audio_files(folder_path: Path):
         audio_exts = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
         return [p for p in folder_path.rglob('*') if p.suffix.lower() in audio_exts]
    
    def add_songs_from_folder(self, folder_path: Path, max_amount: int = 0) -> int:
        files = self.get_audio_files(folder_path)
        if max_amount and max_amount > 0:
            files = files[:max_amount]

        # заранее резервируем id под каждый файл
        tasks: list[tuple[Path, int]] = []
        for p in files:
            song_id = self.db.reserve_song_id()
            tasks.append((Path(p), song_id))

        cpu_count = os.cpu_count() or 1
        max_workers = max(1, int(cpu_count))
        added = 0
        futures=[]
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for file_path,song_id in tasks:
                if self.is_path_exists(file_path):
                    log(f"File already processed, skipping...")
                    continue
                log(f"Submitting {file_path}")
                futures.append(ex.submit(compute_payload,file_path,song_id))

            for fut in as_completed(futures):
                try:
                    payload = fut.result()
                except Exception as e:
                    log(f"Worker error: {e}")
                    continue

                metadata = pp.extract_metadata(file_path, self.metadata)
                fingerprints = payload["fingerprints"]
                file_path = Path(payload["file_path"])
                song_id = payload["song_id"]
                try:
                    self.db.add_song(
                        song_id=song_id,
                        title=metadata.get("title", ""),
                        artist=metadata.get("artist", ""),
                        genre=metadata.get("genre", ""),
                        year=metadata.get("year", ""),
                        album=metadata.get("album", ""),
                        file_path=file_path,
                        fingerprints=fingerprints,
                        duration=metadata.get("duration", '0.0'),
                        save_after=False
                    )
                    log(f"Added {file_path}")
                    added += 1
                except TypeError as e :
                    log(f"ERROR DURING ADDING FILE={e},song_id={song_id},title={metadata.get('title')},file={file_path},dur={metadata.get('duration')}")

                

        # сохраняем один раз после завершения обработки
        self.db.save_all()
        return added






    def add_song_from_file(self, file_path: Path, save_after: bool = True) -> bool:
        file_path = Path(file_path)

        if self.is_path_exists(file_path):
            log(f"File already processed, skipping: {file_path}")
            return False

        song_id = self.db.reserve_song_id()

        audio = pp.load_audio(file_path)
        metadata = pp.extract_metadata(file_path, self.metadata)

        spectrogram = ff.stft(audio)
        points = ff.filter_spectrogram(np.abs(spectrogram).T)
        fingerprints = fp.create_fingerprints(points, song_id)

        self.db.add_song(
            song_id=song_id,
            title=metadata.get("title", file_path.stem),
            artist=metadata.get("artist", ""),
            genre=metadata.get("genre", ""),
            year=metadata.get("year", ""),
            album=metadata.get("album", ""),
            file_path=file_path,
            fingerprints=fingerprints,
            duration=metadata.get("duration", 0.0),
            save_after=False,
        )

        if save_after:
            self.db.save_all()

        return True
        
    def clear_all(self):
        self.db.clear_all()

    def debug_search(self):
        files = self.get_audio_files(cfg.SONGS_DIR)
        for file_path in files:  # итерация по объектам Path
            
            song_id = self.search_song_from_file(str(file_path))
            print(f"{file_path.stem}|{self.get_song_by_id(song_id)['title']}")

        self.db.save_all()
        
    def search_song(self, audio, _debug_correct_id: int = -1, file_path = "") -> Tuple[int, float]:
        try:
            spectrogram = ff.stft(audio)
        except ValueError as e:
            log(f"STFT error: {e}")
            log(f"File path:    {file_path}")
            return -1, -1.
        points = ff.filter_spectrogram(np.abs(spectrogram).T)

        fingerprints_record = fp.create_fingerprints(points, song_id=invalidval)
        query_fp_count = len(fingerprints_record)

        if query_fp_count == 0:
            return -1, -1.0

        addresses: List[int] = [
            address
            for address, _ in fingerprints_record
        ]

        found_fp_list = self.db.fingerprints.lookup_flat_batch(addresses)

        if not found_fp_list:
            return -1, -1.0

        min_matches_per_song = max(5, int(query_fp_count * 0.01))

        id_fps = mf.filter(
            found_fp_list,
            min_matches_per_song=min_matches_per_song,
        )

        if not id_fps:
            return -1, -1.0

        results = mf.analyze_time_coherency(fingerprints_record, id_fps)

        match_id, time_offset, score = self._select_best_match(
            results=results,
            query_fp_count=query_fp_count,
        )

        if match_id == -1:
            return -1, -1.0

        if _debug_correct_id != -1 and match_id == _debug_correct_id and False:
            def _save_to_file(audio_data, output_file: Path, sr: int = cfg.SAMPLE_RATE):
                sf.write(output_file, audio_data, sr, subtype="FLOAT")

            _save_to_file(
                audio_data=audio,
                output_file=Path(cfg.BASE_DIR / f"{_debug_correct_id}.wav"),
            )

        return match_id, -cfg.BIN_TIME * time_offset

    def search_song_from_file(self, file_path: Path) -> int:
        """
        Ищет песню по аудио файлу, возвращает song_id или -1 если не найдено
        """
       #  Обработка аудио 
        audio = pp.load_audio(file_path)
        song_id, offset = self.search_song(audio)
        return song_id
        
       
    def get_random_song(self, rng = np.random.default_rng()):
        return self.db.get_random_song(rng)
    
    def get_song_by_id(self,song_id:int):
        return self.db.get_song_by_id(song_id)

    def print_song_info(self, song_id: int) -> None:
        """
        Печатает информацию о песне
        """
        self.db.print_song_info(song_id)

    def print_all_songs(self) -> None:
        """
        Печатает информацию о всех песнях в базе
        """
        for song_id in self.db.songs.db.keys():
            self.print_song_info(song_id)
            log("-" * 20)    
    
    def close(self):
        """Закрытие и сохранение"""
        self.db.save_all()

    def print_stats(self):
        self.db.print_stats()


    def is_path_exists(self, file_path:Path):
        return self.db.is_path_exists(file_path)
    
    def _select_best_match(
        self,
        results: dict[int, tuple[int, int]],
        query_fp_count: int,
        min_offset_peak: int = 4,
        min_score: float = 0.02,
        min_margin: float = 0.005,
    ) -> tuple[int, int, float]:
        """
        Выбирает лучший результат после analyze_time_coherency.

        Args:
            results:
                Словарь {song_id: (max_count, time_offset)}.
                max_count — максимальное количество совпадений при одном временном сдвиге.
                time_offset — сдвиг в бинах.
            query_fp_count:
                Количество fingerprints в запросе.
            min_offset_peak:
                Минимальное количество совпадений в лучшем offset-пике.
            min_score:
                Минимальная доля совпавших fingerprints запроса.
            min_margin:
                Минимальный разрыв между score лучшего и второго кандидата.

        Returns:
            (song_id, time_offset, score) или (-1, 0, 0.0), если совпадение не принято.
        """
        if not results or query_fp_count <= 0:
            return -1, 0, 0.0

        candidates: list[tuple[float, int, int, int]] = []

        for song_id, (max_count, time_offset) in results.items():
            if max_count < min_offset_peak:
                continue

            score = max_count / query_fp_count

            if score < min_score:
                continue

            candidates.append((score, song_id, max_count, time_offset))

        if not candidates:
            return -1, 0, 0.0

        candidates.sort(reverse=True)

        best_score, best_song_id, _, best_time_offset = candidates[0]

        if len(candidates) > 1:
            second_score = candidates[1][0]

            if best_score - second_score < min_margin:
                return -1, 0, 0.0

        return best_song_id, best_time_offset, best_score