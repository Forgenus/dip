from collections import defaultdict
from typing import Dict, List, Tuple

from config import DELTA_BUCKET_SIZE
from ..processing import fingerprint as fp
def filter(
    found_fp_list: List[Tuple[int, int]],
    min_matches_per_song = -1,
) -> Dict[int, Dict[int, List[int]]]:
    """
    Filters raw DB matches by candidate song.
    """
    if min_matches_per_song == -1:
        min_matches_per_song = max(3, int(len(found_fp_list) * 0.005))
    song_addresses = defaultdict(set)

    for address, hash_value in found_fp_list:
        _, song_id = fp.decode_hash(hash_value)
        song_addresses[song_id].add(address)

    valid_song_ids = {
        song_id
        for song_id, addresses in song_addresses.items()
        if len(addresses) >= min_matches_per_song
    }

    grouped: Dict[int, Dict[int, List[int]]] = {}

    for address, hash_value in found_fp_list:
        anchor_time, song_id = fp.decode_hash(hash_value)

        if song_id not in valid_song_ids:
            continue

        grouped.setdefault(song_id, {})
        grouped[song_id].setdefault(address, [])
        grouped[song_id][address].append(anchor_time)

    return grouped




def analyze_time_coherency(
    record_fingerprints: List[Tuple[int,int]],
    id_fps: Dict[int, Dict[int, List[int]]]
) -> Dict[int, Tuple[int, int]]:
    """
    Анализирует временную согласованность отпечатков между запросом и записями в БД

    Args:
        record_fingerprints: список отпечатков запроса [(address, hash)]
        id_fps: словарь {song_id: {address: List[hash]}}
    
    Returns:
        results: словарь {song_id: (max_count, best_shift)}, где
            max_count — максимальное количество совпадающих отпечатков,
            best_shift — смещение фрагмента относительно записи в бинах
    """
    results: Dict[int, Tuple[int, int]] = {}

    # Декодируем anchor из хэша запроса
    record_times: List[Tuple[int, int]] = [
        (address, fp.decode_hash(hash_value)[0])  
        for address, hash_value in record_fingerprints
    ]

    for song_id, address_map in id_fps.items():
        delta_bins: Dict[int, int] = defaultdict(int)

        for rec_address, rec_anchor_time in record_times:
            if rec_address not in address_map:
                continue

            for song_anchor_time in address_map[rec_address]:
                delta = rec_anchor_time - song_anchor_time
                bucket = round(delta / DELTA_BUCKET_SIZE) * DELTA_BUCKET_SIZE
                delta_bins[bucket] += 1

        if delta_bins:
            max_count = max(delta_bins.values())
            best_shift = next(bucket for bucket, count in delta_bins.items() if count == max_count)
            if not delta_bins:
                continue
            results[song_id] = (max_count, best_shift)


    return results



        


