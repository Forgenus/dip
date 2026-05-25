"""Facade for unified database access"""
from typing import Dict, List, Optional, Tuple, Any, Iterable
from .fingerprint_db import FingerprintDB
from .song_info_db import SongInfoDB
from pathlib import Path
import numpy as np
import threading

log = print
class MusicDatabase:
    """
    Фасад для работы с обеими базами данных
    Объединяет FingerprintDB и SongInfoDB в единый интерфейс
    """
    
    def __init__(self,db_path:Path, fp_name:str="fingerprints", songs_name:str="songs"):
        """
        Инициализация фасада баз данных
        
        Args:
            fp_name: имя для базы отпечатков
            songs_name: имя для базы песен
        """
        self._lock = threading.Lock()
        self.fingerprints = FingerprintDB(fp_name)
        self.songs = SongInfoDB(songs_name)
        self.db_path = db_path
    def add_song(self, song_id: int, title: str, artist: str, genre:str,
                 year:str,album:str,duration:str, file_path: Path, 
                 fingerprints: List[Tuple[int, int]], save_after: bool = True) -> int:
        """
        Добавляет песню и её отпечатки
        
        Args:
            title: название песни
            artist: исполнитель
            file_path: путь к файлу
            fingerprints: список пар Arr[address, hash]
        
        Returns:
            song_id: ID добавленной песни
        """
        # Добавляем информацию о песне
        with self._lock:
            self.songs.add_song(
            song_id=song_id,
            title=title,
            artist=artist,
            file_path=file_path,
            genre=genre,
            year=year,
            duration=float(duration),
            album=album,
            fingerprint_count=len(fingerprints)
        )
            # Добавляем отпечатки в FingerprintDB
            # В hash уже закодированы song_id и время, поэтому просто сохраняем
            for address, hash_value in fingerprints:
                self.fingerprints.insert(address, hash_value)
            
            return song_id
    def reserve_song_id(self) -> int:
        """
        Резервирует новый song_id
        """
        with self._lock:
            return self.songs.reserve_song_id()
    
    def add_songs_batch(self, songs_data: List[Dict[str, Any]]) -> None:
        """
        Добавляет несколько песен пачкой
        
        Args:
            songs_data: список словарей с ключами:
                - title, artist, file_path, fingerprints
        """
        with self._lock:
            for song in songs_data:
                self.add_song(
                    song_id=song['song_id'],
                    title=song.get('title', ''),
                    genre=song.get('genre', ''),
                    year=song.get('year', ''),
                    album=song.get('album', ''),
                    artist=song.get('artist', ''),
                    file_path=song['file_path'],
                    fingerprints=song['fingerprints'],
                    duration=song.get('duration',0.0)
                )

    
    def find_matches(self, query_addresses: List[int]) -> List[Tuple[int, List[int]]]:
        """
        Ищет совпадения для запроса
        
        Args:
            query_addresses: список адресов из запроса
        
        Returns:
            List[address, List[hash]] - совпадения по каждому адресу
        """
        matches: List[Tuple[int, List[int]]] = []
        # Получаем все хэши для каждого адреса
        lookup_results = self.fingerprints.lookup_batch(query_addresses)
        for address, hashes in lookup_results:
            if hashes:
                matches.append((address, hashes))
        return matches
    

    
    def get_song_by_id(self, song_id: int) -> Optional[Dict[str,Any]]:
        """Получает информацию о песне по ID"""
        return self.songs.get_song(song_id)
    
    def get_fingerprints_by_address(self, address: int) -> Tuple[int, List[int]]:
        """Получает хэши по адресу"""
        return self.fingerprints.lookup(address)
    

    
    def clear_all(self) -> None:
        """Очищает обе базы данных"""
        with self._lock:
            self.fingerprints.clear()
            self.songs.clear()
    
    def size(self) -> Dict[str, int]:
        """Возвращает размеры обеих баз"""
        return {
            'fingerprints': self.fingerprints.size(),
            'songs': self.songs.size(),
            'unique_addresses': self.fingerprints.unique_addresses()
        }
    
    def save_all(self) -> None:
        """Сохраняет обе базы"""
        with self._lock:
            fp_changed = self.fingerprints.save(self.db_path)
            songs_changed = self.songs.save(self.db_path)
            if fp_changed != songs_changed:
                log("Warning: FingerprintDB and SongInfoDB have different change states!")
                raise Exception("Inconsistent database state: one changed, other not.")
    
    def load_all(self) -> None:
        """Загружает обе базы"""
        with self._lock:
            self.fingerprints.load(self.db_path)
            self.songs.load(self.db_path)
    
    def print_stats(self) -> None:
        """Выводит статистику по обеим базам"""
        self.fingerprints.print_stats()
        self.songs.print_stats()


    def print_songs(self) ->None:
        """Выводит информацию о всех песнях"""
        songs = self.songs.get_all_songs()
        for song in songs:
            log(song) 


    def print_song_info(self,song_id:int):
        """
        Печатает информацию о песне
        """
        song_info = self.songs.get_song(song_id)
        if not song_info:
            log(f"Song with ID {song_id} not found")
            return
        
        log(f"ID: {song_info['song_id']}")
        log(f"Title: {song_info['title']}")
        log(f"Artist: {song_info['artist']}")
        log(f"Album: {song_info['album']}")
        log(f"Year: {song_info['year']}")
        log(f"Genre: {song_info['genre']}")
        log(f"Duration: {song_info['duration']} seconds")
        log(f"File Path: {song_info['file_path']}")
        log(f"FPs: {song_info['fingerprint_count']}")

    def get_random_song(self, rng)->Dict[str,Any]:
        return self.songs.get_random_song(rng)
    
    def is_path_exists(self, file_path:Path):
        with self._lock:
            return file_path in self.songs.song_paths

    def lookup_flat_batch(self, addresses: Iterable[int]) -> List[Tuple[int, int]]:
        return self.fingerprints.lookup_flat_batch(addresses=addresses)