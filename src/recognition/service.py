"""Service layer for music recognition."""

from pathlib import Path
from typing import Tuple
import json
import logging

import numpy as np

import config as cfg
from ..database import music_database as DB
from ..processing import preprocess as pp
from .batch_indexer import BatchIndexer
from .indexing import compute_payload
from .query_pipeline import QueryPipeline
from .search_trace import SearchTrace


logger = logging.getLogger(__name__)


class MusicRecognitionService:
    """Coordinates processing and database access for music recognition."""

    def __init__(
        self,
        fp_db_name: str = cfg.FINGERPRINT_DB_NAME,
        songs_db_name: str = cfg.SONG_INFO_DB_NAME,
        db_path: Path = cfg.DATABASE_DIR,
    ) -> None:
        self.db = DB.MusicDatabase(db_path, fp_db_name, songs_db_name)
        self.query_pipeline = QueryPipeline(self.db)
        self.metadata = self._load_metadata()
        self.last_search_trace: SearchTrace | None = None
        try:
            self.db.load_all()
        except FileNotFoundError:
            logger.info("No existing database found, starting fresh")

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
        audio_exts = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
        return [p for p in Path(folder_path).rglob("*") if p.suffix.lower() in audio_exts]

    def add_songs_from_folder(self, folder_path: Path, max_amount: int = 0) -> int:
        logger.info("[main] scan folder start: %s", folder_path)
        files = self.get_audio_files(folder_path)
        return BatchIndexer(self.db, self.metadata).add_songs_from_folder(
            Path(folder_path),
            files,
            max_amount=max_amount,
        )

    def add_song_from_file(self, file_path: Path, save_after: bool = True) -> bool:
        file_path = Path(file_path)

        if self.is_path_exists(file_path):
            logger.info("File already processed, skipping: %s", file_path)
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
        for file_path in files:
            song_id, _ = self.search_song_from_file(file_path)
            song = self.get_song_by_id(song_id)
            title = song["title"] if song else "Unknown"
            print(f"{file_path.stem}|{title}")

        self.db.save_all()

    def search_song(
        self,
        audio,
        _debug_correct_id: int = -1,
        file_path: Path | None = None,
        offset_fallback: bool = True,
    ) -> Tuple[int, float]:
        result = self.query_pipeline.search(
            audio,
            expected_id=_debug_correct_id,
            file_path=file_path,
        )

        if offset_fallback and self._is_weak_search_result(result):
            result = self._search_with_offset_fallbacks(
                audio=audio,
                primary_result=result,
                expected_id=_debug_correct_id,
                file_path=file_path,
            )

        self.last_search_trace = result.trace
        return result.song_id, result.time_offset

    def _is_weak_search_result(self, result) -> bool:
        if result.song_id == -1:
            return True

        score = getattr(result.trace, "selected_score", 0.0)
        return score < cfg.SEARCH_OFFSET_FALLBACK_MIN_SCORE

    def _search_with_offset_fallbacks(
        self,
        audio,
        primary_result,
        expected_id: int,
        file_path: Path | None,
    ):
        audio_array = np.asarray(audio)
        best_result = primary_result
        attempts = [self._offset_attempt_record(0, primary_result)]

        for sample_offset in cfg.SEARCH_OFFSET_FALLBACK_SAMPLES:
            if sample_offset <= 0 or sample_offset >= len(audio_array):
                continue

            candidate = self.query_pipeline.search(
                audio_array[sample_offset:],
                expected_id=expected_id,
                file_path=file_path,
            )
            candidate.trace.attempt_sample_offset = sample_offset
            attempts.append(self._offset_attempt_record(sample_offset, candidate))

            if self._is_better_search_result(candidate, best_result):
                candidate.time_offset -= sample_offset / cfg.SAMPLE_RATE
                candidate.trace.offset_fallback_selected = True
                best_result = candidate

        best_result.trace.offset_fallback_attempts = attempts
        return best_result

    def _is_better_search_result(self, candidate, current) -> bool:
        if candidate.song_id == -1:
            return False
        if current.song_id == -1:
            return True

        candidate_score = getattr(candidate.trace, "selected_score", 0.0)
        current_score = getattr(current.trace, "selected_score", 0.0)
        if candidate_score != current_score:
            return candidate_score > current_score

        return getattr(candidate.trace, "selected_max_count", 0) > getattr(
            current.trace,
            "selected_max_count",
            0,
        )

    def _offset_attempt_record(self, sample_offset: int, result) -> dict:
        return {
            "sample_offset": sample_offset,
            "song_id": result.song_id,
            "score": getattr(result.trace, "selected_score", 0.0),
            "max_count": getattr(result.trace, "selected_max_count", 0),
        }

    def search_song_from_file(
        self,
        file_path: Path,
        offset_fallback: bool = True,
    ) -> Tuple[int, float]:
        audio = pp.load_audio(file_path)
        return self.search_song(audio, file_path=file_path, offset_fallback=offset_fallback)

    def get_random_song(self, rng=np.random.default_rng()):
        return self.db.get_random_song(rng)

    def get_song_by_id(self, song_id: int):
        return self.db.get_song_by_id(song_id)

    def print_song_info(self, song_id: int) -> None:
        self.db.print_song_info(song_id)

    def print_all_songs(self) -> None:
        for song_id in self.db.songs.db.keys():
            self.print_song_info(song_id)
            logger.info("-" * 20)

    def close(self):
        self.db.save_all()

    def print_stats(self):
        self.db.print_stats()

    def is_path_exists(self, file_path: Path):
        return self.db.is_path_exists(file_path)
