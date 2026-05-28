"""Song information database"""
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
class SongInfoDB:
    """
    База данных для хранения информации о песнях
    Ключ: song_id (int, auto-increment)
    Значение: словарь с информацией о песне
    """
    
    def __init__(self, name:str="song_info_db"):
        """
        Инициализация базы данных песен
        
        Args:
            name: имя базы (для сохранения/загрузки)
        """
        self.song_paths:set[Path] = set()
        self.changed = False
        self.name = name
        self.db: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        self.stats = {
            'total_songs': 0,
            'total_duration' : 0.0
        }
    def reserve_song_id(self) -> int:
        """
        Резервирует новый song_id
        """
        song_id = self.next_id
        self.next_id += 1
        return song_id
    def add_song(self, song_id: int, file_path: Path,title: str, artist: str = "", genre:str = "",
                  year:str = "", album:str = "", fingerprint_count:int = 0,
                 duration: float = 0.0, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Добавляет новую песню, возвращает song_id
        """
        if song_id in self.db:
            raise ValueError(f"Song with ID {song_id} already exists")
        self.next_id = max(self.next_id, song_id + 1)
    

        song_info:Dict[str, Any] = {
            'song_id': song_id,
            'title': title,
            'artist': artist,
            'genre': genre,
            'year': year,
            'album': album,
            'file_path': file_path,
            'duration': duration,
            'metadata': metadata or {},
            'fingerprint_count' : fingerprint_count
        }
        self.song_paths.add(file_path)
        self.stats['total_duration'] += duration
        self.changed = True
        self.db[song_id] = song_info
        self._update_stats()
        return song_id
    
    def get_song(self, song_id: int) -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию о песне по ID
        """
        return self.db.get(song_id)
    
    def get_songs_batch(self, song_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Возвращает информацию о нескольких песнях
        """
        return {sid: self.db[sid] for sid in song_ids if sid in self.db}
    
    # def update_song(self, song_id: int, **kwargs:dict[str,Any]) -> bool:
    #     """
    #     Обновляет информацию о песне
    #     """
    #     self.changed = True
    #     if song_id not in self.db:
    #         return False
        
    #     self.db[song_id].update(kwargs)
    #     return True
    
    # def remove_song(self, song_id: int) -> bool:
    #     """
    #     Удаляет песню
    #     """
    #     self.changed = True
    #     if song_id in self.db:
    #         del self.db[song_id]
    #         self._update_stats()
    #         return True
    #     return False
    
    def clear(self) -> None:
        """
        Очищает базу данных песен
        """
        self.changed = True
        self.db.clear()
        self.song_paths.clear()
        self.next_id = 0
        self.stats['total_songs'] = 0
        self.stats['total_duration'] = 0
        self._update_stats()
    
    def size(self) -> int:
        """
        Возвращает количество песен
        """
        return len(self.db)
    
    def get_all_songs(self) -> List[Dict[str, Any]]:
        """
        Возвращает список всех песен
        """
        return list(self.db.values())
    
    def get_id_range(self) -> Tuple[int, int]:
        """
        Возвращает диапазон используемых ID
        """
        if self.next_id == 0:
            return (0, 0)
        return (0, self.next_id - 1)
    
    def _update_stats(self) -> None:
        """Обновляет статистику"""
        self.stats['total_songs'] = len(self.db)
    
    def print_stats(self) -> None:
        """Выводит статистику базы"""
        logger.info("\n=== SongInfoDB stats '%s' ===", self.name)
        logger.info("Total songs: %s", self.stats['total_songs'])
        logger.info("Total duration: %s", self.stats['total_duration'])    
    def save(self, path: Path) -> bool:
        """
        Сохраняет базу в файл
        """
        if not self.changed:
            logger.info("No changes to save for SongInfoDB")
            return False
        filename = path / f"{self.name}.pkl"
        data: Dict[str, Any] = {
            'name': self.name,
            'db': self.db,
            'next_id': self.next_id,
            'stats': self.stats,
            'song_paths':self.song_paths
        }
        
        with filename.open('wb') as f:  
            pickle.dump(data, f)
        logger.info("SongInfoDB saved to %s", filename)
        self.changed = False
        return True
    
    def load(self, path: Path) -> None:
        """
        Загружает базу из файла
        """
        filename = path / f"{self.name}.pkl"
        
        if not filename.exists():
            raise FileNotFoundError(f"Файл {filename} не найден")
        
        with filename.open('rb') as f:
            data = pickle.load(f)
        
        self.name = data['name']
        self.db = data['db']
        self.next_id = max(data.get('next_id', 0), max(self.db.keys(), default=-1) + 1)
        self.stats = data.get('stats', {'total_songs': len(self.db)})
        self.song_paths = data.get('song_paths', {song['file_path'] for song in self.db.values()})
        logger.info("SongInfoDB loaded from %s", filename)
        
    def _ensure_no_duplicate_ids(self) -> None:
        """Проверяет, что нет дублирующихся ID"""
        ids = set()
        for song_id in self.db.keys():
            if song_id in ids:
                raise ValueError(f"Duplicate song_id found: {song_id}")
            ids.add(song_id)

    def get_random_song(self, rng) -> Optional[Dict[str, Any]]:
        if not self.db:
            return None

        song_ids = list(self.db.keys())
        index = int(rng.integers(0, len(song_ids)))
        return self.db[song_ids[index]]
