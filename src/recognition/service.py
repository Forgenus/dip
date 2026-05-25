"""
Сервисный слой
"""

from logging import NullHandler
from pathlib import Path
import profile
import sys
from typing import List, Any
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

def compute_payload(file_path:Path,song_id:int) -> dict[str, Any]:
        audio, metadata = pp.load_audio_with_metadata(file_path)
        spectrogram = ff.stft(audio)
        # Создание отпечатков 
        points = ff.filter_spectrogram(np.abs(spectrogram).T)
        fingerprints = fp.create_fingerprints(points, song_id)
        return {
        "file_path": file_path,
        "metadata": metadata,
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
        print(profile)
        self.db = DB.MusicDatabase(db_path, fp_db_name, songs_db_name)
        metadata_path = cfg.SONGS_DIR / "metadata.json"
        existing_metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)

        # Пытаемся загрузить существующие базы
        try:
            self.db.load_all()
        except FileNotFoundError:
            log("No existing database found, starting fresh")


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

        max_workers = int(os.cpu_count()*0.7) or 4
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

                metadata = payload["metadata"] or {}
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
                except TypeError as e :
                    log(f"ERROR DURING ADDING FILE={e},song_id={song_id},title={metadata.get('title')},file={file_path},dur={metadata.get('duration')}")

                added += 1

        # сохраняем один раз после завершения обработки
        self.db.save_all()
        return added






    def add_song_from_file(self, file_path: Path, title: str = '', artist: str = '',
                            genre: str = '', year: str = '', album: str = '', save_after: bool = True) -> bool:
        return False
        
    def clear_all(self):
        self.db.clear_all()

    def debug_search(self):
        files = self.get_audio_files(cfg.SONGS_DIR)
        for file_path in files:  # итерация по объектам Path
            
            song_id = self.search_song_from_file(str(file_path))
            print(f"{file_path.stem}|{self.get_song_by_id(song_id)['title']}")

        self.db.save_all()
    def search_song(self, audio, _debug_correct_id = -1) ->Tuple[int,float]:


        spectrogram = ff.stft(audio)
        points = ff.filter_spectrogram(np.abs(spectrogram).T)
        fingerprints_record = fp.create_fingerprints(points,song_id=invalidval) 
        addresses:List[int] = []
        for address, _ in fingerprints_record:
            addresses.append(address)
        record_targetzones = len(fingerprints_record) -5
        found_fp_list = self.db.fingerprints.lookup_flat_batch(addresses)
       #print(f"Found {len(found_fp_list)} matching fingerprints in DB for query")
        #print(f"Record has {len(fingerprints_record)} fingerprints, target zones: {record_targetzones}")
        id_fps, _debug_cut_correct = mf.filter(found_fp_list, record_targetzones,_debug_correct_id)
        #print(f"id_fps len = {len(id_fps)}")
        results = mf.analyze_time_coherency(fingerprints_record, id_fps)
        if not results:
            return (-1,-1)

        best_score = -1
        match_id = None
        time_offset = None
        for song_id, (max_count, time_offset_curr) in results.items():
            song = self.db.get_song_by_id(song_id)
            try:
                fingerprint_count = int(song['fingerprint_count'])
            except TypeError as e:
                log(f"id={song['song_id']},dur={song['duration']},fps={song['fingerprints']}")
            score = max_count / record_targetzones
            if score > best_score:
                best_score = score
                match_id = song_id
                time_offset = time_offset_curr

        #print(best_score)
        times_matched = results[match_id]
        for song_id, count in results.items():
            entries = self.db.fingerprints.get_entries_by_id(song_id)
            song_name = self.get_song_by_id(song_id)['title']
           # print(f"Song ID {song_id} ({song_name}) has time-coherent matches: {count}, coeff={count/record_targetzones:.2f}, total entries={entries}")
        #log(f"Found match: song_id={match_id}, time_matches={times_matched}, coeff={times_matched/record_targetzones:.2f}")
        if _debug_cut_correct:
            def _save_to_file(audio, output_file: Path, sr: int = cfg.SAMPLE_RATE):
                sf.write(output_file, audio, sr, subtype='FLOAT')
            _save_to_file(audio=audio,output_file=Path(cfg.BASE_DIR / f"{_debug_correct_id}.wav"))
       # print(f"OFFSET = {time_offset}")
        return (match_id,-cfg.BIN_TIME*time_offset) 
    def search_song_from_file(self, file_path: Path) -> int:
        """
        Ищет песню по аудио файлу, возвращает song_id или -1 если не найдено
        """
       #  Обработка аудио 
        audio = pp.load_audio(file_path)
        return self.search_song(audio)
        
       
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
