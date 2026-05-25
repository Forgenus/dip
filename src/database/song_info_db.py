"""Song information database"""
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy
log = print
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
        log(f"\n=== SongInfoDB stats '{self.name}' ===")
        log(f"Total songs: {self.stats['total_songs']}")
        log(f"Total duration: {self.stats['total_duration']}")    
    def save(self, path: Path) -> bool:
        """
        Сохраняет базу в файл
        """
        if not self.changed:
            log("No changes to save for SongInfoDB")
            return False
            return
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
        log(f"SongInfoDB saved to {filename}")
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
        self.next_id = data.get('next_id', 0)
        self.stats = data.get('stats', {'total_songs': len(self.db)})
        self.song_paths = data['song_paths']
        log(f"SongInfoDB loaded from{filename}")

    def get_random_song(self,rng):
        while True:
            song = self.get_song(rng.integers(0,self.next_id-1))
            if song:
                return song
