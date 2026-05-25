from collections import defaultdict
from typing import Dict, List, Tuple
from ..processing import fingerprint as fp
def filter(found_fp_list: List[Tuple[int, int]], record_targetzones: int, _debug_correct_id: int = -1) -> Tuple[Dict[int, Dict[int, List[int]]], bool]:
    """
    Фильтрует песни с недостаточным количеством совпадений

    Args:
        found_fp_list: List[Tuple[address, hash]]
        record_targetzones: приближённое количество отпечатков в запросе
    Returns:
        id_fps: Dict[song_id, Dict[address, List[Tuple[anchor_time, delta]]]]
        _debug_cut_correct: True если правильная песня была отрезана фильтром
    """
    _debug_cut_correct = False

    entries: Dict[int, int] = {}
    for address, hash_value in found_fp_list:
        entries[hash_value] = entries.get(hash_value, 0) + 1

    keys_to_delete: List[int] = [h for h, count in entries.items() if count < 3]
    for h in keys_to_delete:
        del entries[h]

    song_entries: Dict[int, int] = {}
    for fp_hash in entries.keys():
        anchor_time, song_id = fp.decode_hash(fp_hash)  #
        song_entries[song_id] = song_entries.get(song_id, 0) + 1

    # TODO: добавить cutoff по record_targetzones * coeff
    valid_hashes = set(entries.keys())

    id_fps: Dict[int, Dict[int, List[int]]] = {}
    for address, hash_value in found_fp_list:
        if hash_value not in valid_hashes:
            continue

        anchor_time, song_id = fp.decode_hash(hash_value)

        # id_fps[song_id][address] = List[anchor_time]
        if song_id not in id_fps:
            id_fps[song_id] = {}
        if address not in id_fps[song_id]:
            id_fps[song_id][address] = []

        id_fps[song_id][address].append(anchor_time)

   # print(id_fps.keys())
    return id_fps, _debug_cut_correct




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
                delta_bins[delta] += 1

        if delta_bins:
            max_count = max(delta_bins.values())
            best_shift = next(delta for delta, count in delta_bins.items() if count == max_count)
            results[song_id] = (max_count, best_shift)
        else:
            results[song_id] = (0, 0)

    return results



        


