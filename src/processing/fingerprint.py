from typing import List, Tuple
import numpy as np
from config  import MIN_TARGET_DELTA, MAX_TARGET_DELTA , VALIDATE
width = 5
invalidval = 2**32-2
def create_fingerprints(points, song_id):
    points = sorted(points, key=lambda p: (p[0], p[1]))
    fingerprints = []

    for anchor_idx, anchor in enumerate(points):
        anchor_time, anchor_freq = anchor

        targets = []

        for target_idx in range(anchor_idx + 1, len(points)):
            target_time, target_freq = points[target_idx]
            delta = target_time - anchor_time

            if delta < MIN_TARGET_DELTA:
                continue

            if delta > MAX_TARGET_DELTA:
                break

            targets.append((target_time, target_freq))

            if len(targets) == 3:
                break

        if len(targets) < 3:
            continue

        delta = targets[-1][0] - anchor_time

        address = encode_address(
            freq_anchor=anchor_freq,
            freq1=targets[0][1],
            freq2=targets[1][1],
            freq3=targets[2][1],
            delta=delta,
        )

        hash_value = encode_hash(anchor_time, song_id)
        fingerprints.append((address, hash_value))

    return fingerprints


def encode_address(
    freq_anchor: int,
    freq1: int,
    freq2: int,
    freq3: int,
    delta: int,
    validate: bool = VALIDATE
) -> int:
    """
    Кодирует в 64-битное значение:
    - freq_anchor (9 бит)
    - freq1 (9 бит)
    - freq2 (9 бит)
    - freq3 (9 бит)
    - delta (28 бит)
    Всего: 64 бита
    """
    freq_anchor = int(freq_anchor)
    freq1 = int(freq1)
    freq2 = int(freq2)
    freq3 = int(freq3)
    delta = int(delta)
    address = 0

    if validate:
        assert 0 <= freq_anchor < 2**9, f"freq_anchor={freq_anchor} >= 512"
        assert 0 <= freq1 < 2**9, f"freq1={freq1} >= 512"
        assert 0 <= freq2 < 2**9, f"freq2={freq2} >= 512"
        assert 0 <= freq3 < 2**9, f"freq3={freq3} >= 512"
        assert 0 <= delta < 2**28, f"delta={delta} >= 268435456"

    address |= (freq_anchor & 0x1FF) << 55  # биты 55-63
    address |= (freq1 & 0x1FF) << 46         # биты 46-54
    address |= (freq2 & 0x1FF) << 37         # биты 37-45
    address |= (freq3 & 0x1FF) << 28         # биты 28-36
    address |= delta & 0x0FFFFFFF            # биты 0-27

    return address


def decode_address(fingerprint: int, validate: bool = VALIDATE):
    """Декодирует 64-битный address обратно в компоненты"""
    fingerprint = int(fingerprint)
    freq_anchor = (fingerprint >> 55) & 0x1FF
    freq1 = (fingerprint >> 46) & 0x1FF
    freq2 = (fingerprint >> 37) & 0x1FF
    freq3 = (fingerprint >> 28) & 0x1FF
    delta = fingerprint & 0x0FFFFFFF  # 28 бит

    if validate:
        assert freq_anchor < 512
        assert freq1 < 512
        assert freq2 < 512
        assert freq3 < 512
        assert delta < 2**28

    return freq_anchor, freq1, freq2, freq3, delta
def encode_hash(anchor_time: int, song_id: int, validate: bool = VALIDATE) -> int:
    """
    Кодирует anchor_time и song_id в 64-битный hash
    
    Битовое распределение:
    - anchor_time: 32 бита (биты 32-63)
    - song_id:     32 бита (биты 0-31)
    """
    anchor_time = int(anchor_time)
    song_id = int(song_id)
    if validate:
        assert 0 <= anchor_time < 2**32, f"anchor_time={anchor_time} вне диапазона 0-4294967295"
        assert 0 <= song_id < 2**32, f"song_id={song_id} вне диапазона 0-4294967295"
    
    hash_value = 0
    hash_value |= (anchor_time & 0xFFFFFFFF) << 32
    hash_value |= (song_id & 0xFFFFFFFF)
    
    return hash_value

def decode_hash(hash_value:int, validate:bool=VALIDATE) -> Tuple[int, int]:
        """Декодирует 64-битный hash в anchor_time и song_id"""
        hash_value = int(hash_value)
        anchor_time = (hash_value >> 32) & 0xFFFFFFFF
        song_id = hash_value & 0xFFFFFFFF
        
        return anchor_time, song_id  
