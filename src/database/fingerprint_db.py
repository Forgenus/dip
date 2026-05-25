"""Fingerprint database for storing address->hash mappings"""
from pathlib import Path
import pickle
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Iterable
from src.processing.fingerprint import decode_address
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
import config as cfg
log = print
class FingerprintDB:
    """
    База данных для хранения отпечатков в хэш-таблице
    Dict[address, List[hash]]
    """
    
    def __init__(self, name:str="fingerprint_db"):
        """
        Инициализация базы данных отпечатков
        
        Args:
            name: имя базы (для сохранения/загрузки)
        """
        self.changed = False
        self.name = name
        # ключ: address (64-bit), значение: список hash (64-bit)
        self.entry_count: Dict[int, int] = {}
        self.db: Dict[int, List[int]] = defaultdict(list)
        self.stats = {
            'total_entries': 0,  # общее количество записей (address -> hash)
            'unique_addresses': 0,
            'max_list_size': 0
        }
    
    def insert(self, address: int, hash_value: int) -> None:
        """
        Вставляет отпечаток (address, hash) в базу
        """
        _, song_id = self.decode_hash(hash_value)
        self.entry_count[song_id] = self.entry_count.get(song_id, 0) + 1
        self.db[address].append(hash_value)
        self._update_stats_on_insert(address)
        self.changed = True
    
    def insert_many(self, address: int, hash_values: List[int]) -> None:
        """
        Вставляет несколько хэшей для одного адреса
        """
        for hash_value in hash_values:
            _, song_id = self.decode_hash(hash_value)
            self.entry_count[song_id] = self.entry_count.get(song_id, 0) + 1
            
        self.db[address].extend(hash_values)
        self._update_stats_on_insert(address, len(hash_values))
        self.changed = True
    
    def insert_batch(self, data: Dict[int, List[int]]) -> None:
        """
        Вставляет словарь {address: [hash, ...]}
        """
        for address, hash_values in data.items():
            self.db[address].extend(hash_values)
        self._update_stats()
        self.changed = True

    
    def lookup(self, address: int) -> Tuple[int, List[int]]:
        """
        Поиск хэшей по адресу -> List[int]
        Возвращает пустой список, если адрес не найден
        """
        return (address, self.db.get(address, []))
    
    def lookup_batch(self, addresses: Iterable[int]) -> List[Tuple[int, List[int]]]:
        """
        Поиск по нескольким адресам -> список кортежей (address, List[hash])
        """
        return [(addr, self.db.get(addr, [])) for addr in addresses]
    
    def lookup_flat_batch(self, addresses: Iterable[int]) -> List[Tuple[int, int]]:
        """Плоский список (адрес, хэш)"""
        result: List[Tuple[int, int]] = []
        unique = {}
        #temp = {}
        for addr in addresses:
            hashes = self.db.get(addr, [])
            unique[addr] = unique.get(addr, 0) + 1
            for h in hashes:
              #  anchor_time, song_id = self.decode_hash(h)
               # freq1, freq2, freq_target, delta1, delta2 = self.decode_address(addr)
                #print(f"freq1={freq1}, freq2={freq2}, freq_target={freq_target}, delta1={delta1}, delta2={delta2}-> anchor_time={anchor_time}, song_id={song_id} ")
              #  temp[song_id] = temp.get(song_id, 0) + 1
             #   if(song_id!=0):
               #     print(f"Found fingerprint with song_id={song_id}")
                result.append((addr, h))
       # print(temp.items())
        #unique_cnt = [key for key,count in unique.items() if count==1]
        #print(f"Unique addresses in query: {len(unique_cnt)}")
        return result
    
    def contains(self, address: int) -> bool:
        """
        Проверяет, есть ли адрес в базе
        """
        return address in self.db
    
    def remove(self, address: int) -> bool:
        """
        Удаляет все записи по адресу
        """
        if address in self.db:
            del self.db[address]
            self._update_stats()
            self.changed = True
            return True
        return False
    
    def clear(self) -> None:
        """
        Очищает базу данных
        """
        self.db.clear()
        self.changed = True
        self._update_stats()
    
    def size(self) -> int:
        """
        Возвращает общее количество записей (address -> hash)
        """
        return sum(len(hash_values) for hash_values in self.db.values())
    
    def unique_addresses(self) -> int:
        """
        Возвращает количество уникальных адресов
        """
        return len(self.db)
    
    def get_all_addresses(self) -> List[int]:
        """
        Возвращает список всех адресов
        """
        return list(self.db.keys())
    
    def get_all_hashes(self) -> List[int]:
        """
        Возвращает все хэши (без разбивки по адресам)
        """
        all_hashes:List[int] = []
        for hashes in self.db.values():
            all_hashes.extend(hashes)
        return all_hashes
    
    def _update_stats_on_insert(self, address: int, count: int = 1) -> None:
        """Обновляет статистику при вставке"""
        self.stats['total_entries'] += count
        self.stats['unique_addresses'] = len(self.db)
        self.stats['max_list_size'] = max(
            self.stats['max_list_size'], 
            len(self.db[address])
        )
    def _print_anchor_times(self):
        for (address, hashes) in self.db.items():
            for hash in hashes:
                (time,id) = self.decode_hash(hash)
                print(f"time={time} id={id}")

    def _update_stats(self) -> None:
        """Пересчитывает всю статистику"""
        self.stats['total_entries'] = self.size()
        self.stats['unique_addresses'] = len(self.db)
        self.stats['max_list_size'] = max(
            (len(hash_values) for hash_values in self.db.values()), 
            default=0
        )
    
    def print_stats(self) -> None:
        """Выводит статистику базы"""
        log(f"\n=== FingerprintDB stats '{self.name}' ===")
        log(f"Total entries : {self.stats['total_entries']:,}")
        log(f"Unique addresses: {self.stats['unique_addresses']:,}")
        log(f"Max list size for one address: {self.stats['max_list_size']}")
        for (address,hashes) in self.db.items():
            if len(hashes) > 1000:
                freq1,freq2, freq_target, delta1, delta2 = decode_address(address)
                print(f"freq1={freq1}freq2={freq2}target={freq_target}delta1={delta1}delta2={delta2}")
        if self.stats['unique_addresses'] > 0:
            avg = self.stats['total_entries'] / self.stats['unique_addresses']
            log(f"Avg list length: {avg:.2f}")
    
    def save(self, path: Path) -> bool:
        """
        Сохраняет базу в файл
        """
        if(not self.changed):
            log("No changes to save for FingerprintDB")
            return False
        filename = os.path.join(path, f"{self.name}.pkl")
        
        data: Dict[str, Any] = {
            'name': self.name,
            'db': dict(self.db),
            'stats': self.stats
        }
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        self.changed = False
        log(f"FingerprintDB saved to {filename}")
        return True
    
    def load(self, path: Path) -> None:
        """
        Загружает базу из файла
        """
        filename = path / f"{self.name}.pkl"
        
        if not filename.exists():
            raise FileNotFoundError(f"File {filename} not found")
        
        log(filename)
        with filename.open('rb') as f:
            data = pickle.load(f)
        
        self.changed = False
        self.name = data['name']
        self.db = defaultdict(list, data['db'])
        self.stats = data['stats']
        log(f"FingerprintDB loaded from {filename}")



    def print_song_entries(self) -> None:
        """Выводит количество записей для каждого song_id"""
        log("\n=== Song ID entry counts ===")
        for song_id, count in self.entry_count.items():
            log(f"Song ID {song_id}: {count} entries")


    def get_entries_by_id(self, song_id: int) -> int:
        """Возвращает количество записей для данного song_id"""
        return self.entry_count.get(song_id, 0)
    @staticmethod
    def decode_hash(hash_value:int, validate:bool=cfg.VALIDATE) -> Tuple[int, int]:
        """Декодирует 64-битный hash в anchor_time и song_id"""
        anchor_time = (hash_value >> 32) & 0xFFFFFFFF
        song_id = hash_value & 0xFFFFFFFF
        
        return anchor_time, song_id         
    