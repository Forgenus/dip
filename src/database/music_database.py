"""Facade for unified database access."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging
import threading

from .fingerprint_db import FingerprintDB
from .song_info_db import SongInfoDB


logger = logging.getLogger(__name__)


class MusicDatabase:
    """Unified access to fingerprint and song-info databases."""

    def __init__(self, db_path: Path, fp_name: str = "fingerprints", songs_name: str = "songs"):
        self._lock = threading.Lock()
        self.fingerprints = FingerprintDB(fp_name)
        self.songs = SongInfoDB(songs_name)
        self.db_path = db_path

    def add_song(
        self,
        song_id: int,
        title: str,
        artist: str,
        genre: str,
        year: str,
        album: str,
        duration: str,
        file_path: Path,
        fingerprints: List[Tuple[int, int]],
        save_after: bool = True,
    ) -> int:
        with self._lock:
            return self._add_song_unlocked(
                song_id=song_id,
                title=title,
                artist=artist,
                genre=genre,
                year=year,
                album=album,
                duration=duration,
                file_path=file_path,
                fingerprints=fingerprints,
            )

    def _add_song_unlocked(
        self,
        song_id: int,
        title: str,
        artist: str,
        genre: str,
        year: str,
        album: str,
        duration: str,
        file_path: Path,
        fingerprints: List[Tuple[int, int]],
    ) -> int:
        self.songs.add_song(
            song_id=song_id,
            title=title,
            artist=artist,
            file_path=file_path,
            genre=genre,
            year=year,
            duration=float(duration),
            album=album,
            fingerprint_count=len(fingerprints),
        )

        for address, hash_value in fingerprints:
            self.fingerprints.insert(address, hash_value)

        return song_id

    def reserve_song_id(self) -> int:
        with self._lock:
            return self.songs.reserve_song_id()

    def add_songs_batch(self, songs_data: List[Dict[str, Any]]) -> None:
        with self._lock:
            for song in songs_data:
                self._add_song_unlocked(
                    song_id=song["song_id"],
                    title=song.get("title", ""),
                    genre=song.get("genre", ""),
                    year=song.get("year", ""),
                    album=song.get("album", ""),
                    artist=song.get("artist", ""),
                    file_path=song["file_path"],
                    fingerprints=song["fingerprints"],
                    duration=song.get("duration", 0.0),
                )

    def find_matches(self, query_addresses: List[int]) -> List[Tuple[int, List[int]]]:
        matches: List[Tuple[int, List[int]]] = []
        lookup_results = self.fingerprints.lookup_batch(query_addresses)
        for address, hashes in lookup_results:
            if hashes:
                matches.append((address, hashes))
        return matches

    def get_song_by_id(self, song_id: int) -> Optional[Dict[str, Any]]:
        return self.songs.get_song(song_id)

    def get_fingerprints_by_address(self, address: int) -> Tuple[int, List[int]]:
        return self.fingerprints.lookup(address)

    def clear_all(self) -> None:
        with self._lock:
            self.fingerprints.clear()
            self.songs.clear()

    def size(self) -> Dict[str, int]:
        return {
            "fingerprints": self.fingerprints.size(),
            "songs": self.songs.size(),
            "unique_addresses": self.fingerprints.unique_addresses(),
        }

    def save_all(self) -> None:
        with self._lock:
            logger.info("Saving FingerprintDB")
            fp_changed = self.fingerprints.save(self.db_path)
            logger.info("Saving SongInfoDB")
            songs_changed = self.songs.save(self.db_path)
            if fp_changed != songs_changed:
                logger.warning("FingerprintDB and SongInfoDB have different change states")
                raise Exception("Inconsistent database state: one changed, other not.")

    def load_all(self) -> None:
        with self._lock:
            self.fingerprints.load(self.db_path)
            self.songs.load(self.db_path)

    def print_stats(self) -> None:
        self.fingerprints.print_stats()
        self.songs.print_stats()

    def print_songs(self) -> None:
        songs = self.songs.get_all_songs()
        for song in songs:
            logger.info(song)

    def print_song_info(self, song_id: int) -> None:
        song_info = self.songs.get_song(song_id)
        if not song_info:
            logger.info("Song with ID %s not found", song_id)
            return

        logger.info("ID: %s", song_info["song_id"])
        logger.info("Title: %s", song_info["title"])
        logger.info("Artist: %s", song_info["artist"])
        logger.info("Album: %s", song_info["album"])
        logger.info("Year: %s", song_info["year"])
        logger.info("Genre: %s", song_info["genre"])
        logger.info("Duration: %s seconds", song_info["duration"])
        logger.info("File Path: %s", song_info["file_path"])
        logger.info("FPs: %s", song_info["fingerprint_count"])

    def get_random_song(self, rng) -> Optional[Dict[str, Any]]:
        return self.songs.get_random_song(rng)

    def is_path_exists(self, file_path: Path) -> bool:
        with self._lock:
            return file_path in self.songs.song_paths

    def lookup_flat_batch(self, addresses: Iterable[int]) -> List[Tuple[int, int]]:
        return self.fingerprints.lookup_flat_batch(addresses=addresses)
