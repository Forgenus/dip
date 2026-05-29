"""Fingerprint database for storing address -> hash mappings."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import logging
import os
import pickle

from src.native._fingerprint_index import FingerprintIndex
from src.processing.fingerprint import decode_address, decode_hash


logger = logging.getLogger(__name__)
BACKEND_NAME = "pybind11_binary_v1"


class FingerprintDB:
    """Stores fingerprints as address -> List[hash] mappings."""

    def __init__(self, name: str = "fingerprint_db"):
        self.name = name
        self.entry_count: Dict[int, int] = {}
        self.db = FingerprintIndex()
        self.changed = False
        self.stats = {
            "total_entries": 0,
            "unique_addresses": 0,
            "max_list_size": 0,
        }

    @property
    def changed(self) -> bool:
        return self.db.changed()

    @changed.setter
    def changed(self, value: bool) -> None:
        self.db.set_changed(value)

    def insert(self, address: int, hash_value: int) -> None:
        _, song_id = decode_hash(hash_value)
        self.entry_count[song_id] = self.entry_count.get(song_id, 0) + 1
        self.db.insert(address, hash_value)
        self._update_stats_on_insert(address)

    def insert_many(self, address: int, hash_values: List[int]) -> None:
        self._update_entry_count(hash_values)
        self.db.insert_many(address, hash_values)
        self._update_stats_on_insert(address, len(hash_values))

    def insert_batch(self, data: Dict[int, List[int]]) -> None:
        for address, hash_values in data.items():
            self.db.insert_many(address, hash_values)
            self._update_entry_count(hash_values)
        self._update_stats()

    def lookup(self, address: int) -> Tuple[int, List[int]]:
        return (address, self.db.lookup(address))

    def lookup_batch(self, addresses: Iterable[int]) -> List[Tuple[int, List[int]]]:
        return [(addr, self.db.lookup(addr)) for addr in addresses]

    def lookup_flat_batch(self, addresses: Iterable[int]) -> List[Tuple[int, int]]:
        return self.db.lookup_flat_batch(list(addresses))

    def contains(self, address: int) -> bool:
        return self.db.contains(address)

    def remove(self, address: int) -> bool:
        if self.db.contains(address):
            self.db.remove(address)
            self._update_stats()
            self._rebuild_entry_count()
            return True
        return False

    def clear(self) -> None:
        self.db.clear()
        self.entry_count.clear()
        self._update_stats()

    def size(self) -> int:
        return self.db.size()

    def unique_addresses(self) -> int:
        return self.db.unique_addresses()

    def get_all_addresses(self) -> List[int]:
        return [address for address, _ in self.db.items()]

    def get_all_hashes(self) -> List[int]:
        all_hashes: List[int] = []
        for _, hashes in self.db.items():
            all_hashes.extend(hashes)
        return all_hashes

    def _update_stats_on_insert(self, address: int, count: int = 1) -> None:
        self.stats["total_entries"] += count
        self.stats["unique_addresses"] = self.db.unique_addresses()
        self.stats["max_list_size"] = max(
            self.stats["max_list_size"],
            len(self.db.lookup(address)),
        )

    def _print_anchor_times(self) -> None:
        for _, hashes in self.db.items():
            for hash_value in hashes:
                anchor_time, song_id = decode_hash(hash_value)
                logger.info("time=%s id=%s", anchor_time, song_id)

    def _update_stats(self) -> None:
        self.stats["total_entries"] = self.size()
        self.stats["unique_addresses"] = self.db.unique_addresses()
        self.stats["max_list_size"] = max(
            (len(hash_values) for _, hash_values in self.db.items()),
            default=0,
        )

    def _update_entry_count(self, hash_values: Iterable[int]) -> None:
        for hash_value in hash_values:
            _, song_id = decode_hash(hash_value)
            self.entry_count[song_id] = self.entry_count.get(song_id, 0) + 1

    def _rebuild_entry_count(self) -> None:
        self.entry_count = {}
        for _, hash_values in self.db.items():
            self._update_entry_count(hash_values)

    def print_stats(self) -> None:
        logger.info("\n=== FingerprintDB stats '%s' ===", self.name)
        logger.info("Total entries : %s", f"{self.stats['total_entries']:,}")
        logger.info("Unique addresses: %s", f"{self.stats['unique_addresses']:,}")
        logger.info("Max list size for one address: %s", self.stats["max_list_size"])
        if self.stats["unique_addresses"] > 0:
            avg = self.stats["total_entries"] / self.stats["unique_addresses"]
            logger.info("Avg list length: %.2f", avg)

    def save(self, path: Path) -> bool:
        if not self.changed and not self.db.changed():
            logger.info("No changes to save for FingerprintDB")
            return False

        data_filename = path / f"{self.name}.fpbin"
        meta_filename = path / f"{self.name}.meta.pkl"
        data: Dict[str, Any] = {
            "name": self.name,
            "backend": BACKEND_NAME,
            "stats": self.stats,
            "entry_count": self.entry_count,
        }
        os.makedirs(path, exist_ok=True)
        self.db.save_binary(str(data_filename))
        with meta_filename.open("wb") as f:
            pickle.dump(data, f)
        self.changed = False
        logger.info("FingerprintDB saved to %s", data_filename)
        return True

    def load(self, path: Path) -> None:
        data_filename = path / f"{self.name}.fpbin"
        meta_filename = path / f"{self.name}.meta.pkl"
        old_pickle_filename = path / f"{self.name}.pkl"

        if not data_filename.exists():
            if old_pickle_filename.exists():
                raise FileNotFoundError(
                    "Fingerprint DB binary file not found. "
                    "Recreate database or migrate old pickle manually."
                )
            raise FileNotFoundError(f"File {data_filename} not found")

        logger.info(data_filename)

        self.changed = False
        self.db = FingerprintIndex()
        self.db.load_binary(str(data_filename))

        if meta_filename.exists():
            with meta_filename.open("rb") as f:
                data = pickle.load(f)
            self.name = data.get("name", self.name)
            self.stats = data.get(
                "stats",
                {
                    "total_entries": self.db.size(),
                    "unique_addresses": self.db.unique_addresses(),
                    "max_list_size": 0,
                },
            )
            self.entry_count = data.get("entry_count", {})
        else:
            self.stats = {
                "total_entries": self.db.size(),
                "unique_addresses": self.db.unique_addresses(),
                "max_list_size": 0,
            }
            self.entry_count = {}

        self.changed = False
        logger.info("FingerprintDB loaded from %s", data_filename)

    def print_song_entries(self) -> None:
        logger.info("\n=== Song ID entry counts ===")
        for song_id, count in self.entry_count.items():
            logger.info("Song ID %s: %s entries", song_id, count)

    def get_entries_by_id(self, song_id: int) -> int:
        return self.entry_count.get(song_id, 0)
