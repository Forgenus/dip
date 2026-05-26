"""
Сервисный слой
"""

from pathlib import Path
from typing import Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json
import numpy as np

from ..database import music_database as DB
from ..processing import fft_filter as ff
from ..processing import preprocess as pp
from ..processing import fingerprint as fp
from . import match_filter as mf
from .indexing import compute_payload
from .search_trace import SearchTrace
from .scoring import compute_candidate_score, select_best_match
import config as cfg

invalidval = 2**32 - 2

log = print

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
        self.last_search_trace: SearchTrace | None = None
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

                file_path = payload.file_path
                song_id = payload.song_id
                metadata = pp.extract_metadata(file_path, self.metadata)
                fingerprints = payload.fingerprints

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
                except TypeError as e:
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

        payload = compute_payload(file_path, song_id)
        metadata = pp.extract_metadata(file_path, self.metadata)

        self.db.add_song(
            song_id=song_id,
            title=metadata.get("title", file_path.stem),
            artist=metadata.get("artist", ""),
            genre=metadata.get("genre", ""),
            year=metadata.get("year", ""),
            album=metadata.get("album", ""),
            file_path=file_path,
            fingerprints=payload.fingerprints,
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
            song_id = self.search_song_from_file(file_path)
            song = self.get_song_by_id(song_id)
            title = song["title"] if song else "Unknown"
            print(f"{file_path.stem}|{title}")

        self.db.save_all()
        
    def search_song(
    self,
    audio,
    _debug_correct_id: int = -1,
    file_path: Path | None = None,
    ) -> Tuple[int, float]:
        trace = SearchTrace(expected_id=_debug_correct_id)
        self.last_search_trace = trace

        try:
            spectrogram = ff.stft(audio)
        except ValueError as e:
            trace.dropped_stage = "stft"
            trace.reason = str(e)
            log(f"STFT error: {e}")
            log(f"File path: {file_path}")
            return -1, -1.0

        points = ff.filter_spectrogram(np.abs(spectrogram).T)

        fingerprints_record = fp.create_fingerprints(
            points,
            song_id=invalidval,
        )

        query_fp_count = len(fingerprints_record)
        trace.query_fp_count = query_fp_count

        if query_fp_count == 0:
            trace.dropped_stage = "fingerprint_creation"
            trace.reason = "query produced 0 fingerprints"
            return -1, -1.0

        addresses: List[int] = [
            address
            for address, _ in fingerprints_record
        ]

        found_fp_list = self.db.fingerprints.lookup_flat_batch(addresses)

        trace.db_match_count = len(found_fp_list)
        trace.correct_in_db_lookup = self._song_id_in_found_matches(
            found_fp_list,
            _debug_correct_id,
        )

        if not found_fp_list:
            trace.dropped_stage = "db_lookup"
            trace.reason = "no matching addresses found in DB"
            return -1, -1.0

        if _debug_correct_id != -1 and not trace.correct_in_db_lookup:
            trace.dropped_stage = "db_lookup"
            trace.reason = "correct song was not returned by fingerprint DB lookup"
            return -1, -1.0

        min_matches_per_song = max(5, int(query_fp_count * 0.01))

        id_fps = mf.filter(
            found_fp_list,
            min_matches_per_song=min_matches_per_song,
        )

        trace.candidates_after_filter = list(id_fps.keys())
        trace.correct_after_filter = _debug_correct_id in id_fps

        if not id_fps:
            trace.dropped_stage = "filter"
            trace.reason = "all candidates were removed by match_filter.filter"
            return -1, -1.0

        if _debug_correct_id != -1 and not trace.correct_after_filter:
            trace.dropped_stage = "filter"
            trace.reason = "correct song was removed by match_filter.filter"
            return -1, -1.0

        results = mf.analyze_time_coherency(
            fingerprints_record,
            id_fps,
        )
        if _debug_correct_id in results:
            expected_max_count, expected_offset = results[_debug_correct_id]
            trace.expected_max_count = expected_max_count
            trace.expected_time_offset = expected_offset
            trace.expected_score = compute_candidate_score(
                expected_max_count,
                query_fp_count,
            )

        trace.candidates_after_time = results.copy()
        trace.correct_after_time = _debug_correct_id in results
        trace.correct_time_result = results.get(_debug_correct_id)

        if not results:
            trace.dropped_stage = "time_coherency"
            trace.reason = "no candidates after analyze_time_coherency"
            return -1, -1.0

        if _debug_correct_id != -1 and not trace.correct_after_time:
            trace.dropped_stage = "time_coherency"
            trace.reason = "correct song was removed by analyze_time_coherency"
            return -1, -1.0

        selection = select_best_match(
            results=results,
            query_fp_count=query_fp_count,
        )

        match_id = selection.song_id
        time_offset = selection.time_offset
        score = selection.score

        trace.selected_id = match_id
        trace.selected_score = score
        trace.reason = selection.reason
        
        if match_id in results:
            trace.selected_max_count = results[match_id][0]
            
        if match_id == -1:
            trace.dropped_stage = "selection"
            trace.reason = selection.reason or "no candidate passed final selection thresholds"
            return -1, -1.0

        if _debug_correct_id != -1 and match_id != _debug_correct_id:
            trace.dropped_stage = "selection"
            trace.reason = (
                "correct song survived previous stages, "
                "but another candidate was selected"
            )
            return match_id, -cfg.BIN_TIME * time_offset

        trace.dropped_stage = "matched"
        trace.reason = "song matched successfully"

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
    def _song_id_in_found_matches(
    self,
    found_fp_list: list[tuple[int, int]],
    expected_id: int,
    ) -> bool:
        if expected_id == -1:
            return False

        for _, hash_value in found_fp_list:
            _, song_id = fp.decode_hash(hash_value)

            if song_id == expected_id:
                return True
        return False
