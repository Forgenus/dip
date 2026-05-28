from collections import defaultdict
from typing import Dict, List, Tuple

from config import DELTA_BUCKET_SIZE
from ..processing import fingerprint as fp


def count_raw_matches_by_song(found_fp_list: List[Tuple[int, int]]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)

    for _, hash_value in found_fp_list:
        _, song_id = fp.decode_hash(hash_value)
        counts[song_id] += 1

    return dict(counts)


def count_unique_addresses_by_song(
    id_fps: Dict[int, Dict[int, List[int]]],
) -> Dict[int, int]:
    return {
        song_id: len(address_map)
        for song_id, address_map in id_fps.items()
    }


def filter(
    found_fp_list: List[Tuple[int, int]],
    min_matches_per_song=2,
) -> Dict[int, Dict[int, List[int]]]:
    """Group raw DB matches by song and address."""
    if min_matches_per_song == -1:
        min_matches_per_song = 0

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


def time_coherency_buckets_for_song(
    record_fingerprints: List[Tuple[int, int]],
    id_fps: Dict[int, Dict[int, List[int]]],
    song_id: int,
) -> Dict[int, int]:
    address_map = id_fps.get(song_id)
    if not address_map:
        return {}

    delta_bins: Dict[int, int] = defaultdict(int)

    for rec_address, hash_value in record_fingerprints:
        if rec_address not in address_map:
            continue

        rec_anchor_time, _ = fp.decode_hash(hash_value)

        for song_anchor_time in address_map[rec_address]:
            delta = rec_anchor_time - song_anchor_time
            bucket = round(delta / DELTA_BUCKET_SIZE) * DELTA_BUCKET_SIZE
            delta_bins[bucket] += 1

    return dict(sorted(delta_bins.items(), key=lambda item: (-item[1], item[0])))


def analyze_time_coherency(
    record_fingerprints: List[Tuple[int, int]],
    id_fps: Dict[int, Dict[int, List[int]]],
) -> Dict[int, Tuple[int, int]]:
    """
    Return {song_id: (max_count, best_shift)} after offset bucketing.
    """
    results: Dict[int, Tuple[int, int]] = {}

    for song_id in id_fps:
        delta_bins = time_coherency_buckets_for_song(
            record_fingerprints,
            id_fps,
            song_id,
        )

        if delta_bins:
            max_count = max(delta_bins.values())
            best_shift = next(
                bucket for bucket, count in delta_bins.items() if count == max_count
            )
            results[song_id] = (max_count, best_shift)

    return results
