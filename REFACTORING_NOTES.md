# Refactoring Notes

## Changed

- Updated recognition score formula:
  - `score = max_count / total_matches`
  - `total_matches = len(found_fp_list)` immediately after `lookup_flat_batch`
  - division by zero is handled: `total_matches <= 0` returns `0.0`
- Kept `query_fp_count` as a separate metric for diagnostics and filtering thresholds.
- Extracted query recognition flow from `MusicRecognitionService.search_song` into `src/recognition/query_pipeline.py`.
- Extracted batch folder indexing from `MusicRecognitionService.add_songs_from_folder` into `src/recognition/batch_indexer.py`.
- Added focused scoring and query-pipeline tests in `tests/test_scoring.py` and `tests/test_query_pipeline.py`.
- Updated trace reporting to print `total_matches` instead of the older `db_match_count` label.

## Files Touched

- `src/recognition/scoring.py`
- `src/recognition/query_pipeline.py`
- `src/recognition/search_trace.py`
- `src/recognition/service.py`
- `src/recognition/batch_indexer.py`
- `src/testing/reporting.py`
- `tests/test_scoring.py`
- `tests/test_query_pipeline.py`
- `REFACTORING_NOTES.md`

## Moved Responsibilities

- Search pipeline moved from `MusicRecognitionService.search_song` to `QueryPipeline.search`.
- Query fingerprint creation moved to `build_query_fingerprints`.
- DB lookup moved to `lookup_matches`.
- Candidate filtering moved to `filter_matches`.
- Expected-result trace update moved to `update_expected_trace`.
- DB lookup debug check moved to `song_id_in_found_matches`.
- Batch folder indexing moved to `BatchIndexer.add_songs_from_folder`.
- Worker future handling, metadata extraction, DB insertion, executor cleanup, and pending-task reporting now live in `BatchIndexer`.

## Automatically Removed

- Removed the meaningless `MusicRecognitionService._select_best_match` wrapper.
- Removed service-level private search helpers that only belonged to the extracted query pipeline.
- Removed imports from `service.py` that became unnecessary after extraction.

## Left For Owner Decision

- `src/database/song_info_db.py`: commented-out `update_song` and `remove_song` methods look like old code.
- `src/recognition/indexing.py`: worker `print(..., flush=True)` calls duplicate logging and may be production noise.
- `src/cli/commands.py`: `print(f"Parsed arguments: {args}")` is probably debug output in normal CLI flow.
- `src/recognition/service.py`: `debug_search` prints directly and may belong in testing/debug tooling instead of service.
- `src/cli/parser.py`: `--noise`, `--noise-level`, `--volume`, and `--volume-factor` are documented as passed through but not applied by `TestRunner`.
- `src/neural/neural.py`: appears experimental/standalone, with interactive input and direct `print` calls.
- `split.py`: standalone script with hardcoded source/destination constants and side effects.
- `src/database/fingerprint_db.py`: `_print_anchor_times` appears debug-only and may be unused.
- Several Russian comments/help strings appear mojibake-encoded in the current files; fix separately to avoid mixing encoding cleanup with behavior refactoring.

## Score Threshold Warning

The score scale changed and will usually be much smaller than before. Existing `min_score` thresholds may require experimental retuning. No new magic threshold values were introduced in this refactor.

## Notes

- `BatchIndexer` keeps the effective previous worker count at `1` to avoid changing indexing behavior during this refactor.
- Batch indexing now keeps a `future -> PendingSong(file_path, song_id)` mapping and logs pending files when worker waiting stalls.
- The repository already had unrelated modified files before this refactor; those were not reverted.
