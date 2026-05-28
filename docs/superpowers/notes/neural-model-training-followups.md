# Neural Model Training Follow-ups

These are non-blocking review notes. They should not stop the current MVP implementation unless they become correctness issues.

## Task 2: Song Split Helpers

- `split_song_items()` could validate duplicate `song_id` values explicitly.
- Small split inputs could have an explicit policy for minimum song count and ratio rounding.
- `_portable_path()` could normalize outside-`BASE_DIR` paths with POSIX separators for more stable JSON.

## Task 3: Neural Split CLI

- Add dedicated CLI/handler tests for overwrite refusal, empty database handling, and summary output.

## Task 4: Symmetric Torchaudio Features

- Update or remove stale `tests/test_neural_features.py` coverage that still imports old `build_pair_features` and `log_mel` APIs.
