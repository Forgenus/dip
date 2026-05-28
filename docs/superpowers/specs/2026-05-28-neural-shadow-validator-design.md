# Neural Shadow Validator Design

## Purpose

Add a neural validation layer to the existing music recognition pipeline without changing search results at first.

The first version runs in shadow mode. It observes fingerprint matches, evaluates candidate audio pairs with a local PyTorch pair classifier, and records diagnostic data. Later, the same foundation can become a gray-zone validator that helps weak matches and rejects false positives.

Primary first-stage goal:

- Collect reliable neural diagnostics without changing the current fingerprint result.

Future goal:

- Improve weak-match recall while keeping false positives from increasing.

## Current Pipeline Context

The current recognition flow is:

1. Build query fingerprints.
2. Look up matching addresses in the fingerprint database.
3. Group and filter matches by song.
4. Analyze time coherency.
5. Select the best match by score and offset peak.
6. Optionally retry with offset fallbacks.

The neural layer does not replace this flow. Fingerprints remain the fast candidate generator. The neural layer validates candidate pairs after the normal pipeline has produced candidate songs and offsets.

## Chosen Approach

Use a local PyTorch pair classifier.

Input:

- Query audio window.
- Candidate song audio window at the fingerprint-estimated offset.
- Fixed target window length: 5 seconds.
- Model input channels:
  1. `query_logmel`
  2. `candidate_logmel`
  3. `abs(query_logmel - candidate_logmel)`

Output:

- `same_probability`: float from 0.0 to 1.0.
- `decision`: `same` or `not_same` using a configurable threshold.
- `reliability`: `high`, `medium`, `low`, or `skipped`.

The model does not expose embeddings as a product concern. It directly answers whether two 5-second fragments appear to be the same match.

## Runtime Modes

### Shadow Wide

Initial mode.

The neural validator runs broadly:

- Successful matches.
- Weak matches.
- Failed matches if candidate data is available.
- Top-N candidates once the pipeline records them.

It never changes the returned `song_id` or `time_offset`.

### Gray Zone Validator

Future mode.

The neural validator runs only when fingerprint confidence is questionable:

- Low score.
- Low `max_count`.
- Small margin between top candidates.
- Offset fallback selected or still weak.
- Candidate survived fingerprint stages but final confidence is poor.

In this future mode it may reject suspicious false positives or confirm weak correct matches.

## Top Candidate Trace

`QueryPipeline` should continue to own fingerprint-only logic, but should record top candidates after time coherency.

Candidate trace shape:

```text
song_id
rank
score
max_count
time_offset_bins
time_offset_seconds
```

The selected result remains unchanged. Top candidates are diagnostic data for neural validation and future reranking.

## Service Integration

`MusicRecognitionService.search_song()` orchestrates the optional neural layer after the normal query pipeline returns.

Flow:

```text
query_pipeline.search(audio)
  -> SearchResult
  -> if neural shadow enabled:
       neural_validator.evaluate_top_candidates(audio, trace.top_candidates)
  -> attach neural results to trace
  -> return original fingerprint result unchanged
```

The neural validator:

1. Receives query audio and top candidates.
2. Loads each candidate song by `song_id`.
3. Crops a 5-second candidate window around `time_offset_seconds`.
4. Crops or pads the query to 5 seconds.
5. Builds the 3-channel log-mel pair tensor.
6. Runs the pair classifier.
7. Returns candidate-level neural results.

Any neural error must be non-fatal in shadow mode. Missing model files, unavailable PyTorch, crop failures, and inference errors should be recorded in trace and must not change the fingerprint result.

## Audio Window Policy

Target neural window:

- 5.0 seconds.

Candidate window:

- Always 5.0 seconds when enough audio is available.
- Crop around the fingerprint-estimated candidate offset.
- If the crop reaches the start or end of the song, pad the missing side with zeros and record the valid duration.

Query window:

- If query is at least 5.0 seconds: use the first 5.0 seconds for MVP. Smarter energy-based selection can be added later.
- If query is 3.0 to 5.0 seconds: zero-pad to 5.0 seconds, reliability `medium`.
- If query is 2.0 to 3.0 seconds: zero-pad to 5.0 seconds, reliability `low`.
- If query is below 2.0 seconds: skip neural validation or log it as skipped.

Padding:

- Use zeros.
- Store actual valid duration and padding ratio.

Trace fields:

```text
neural_window_seconds
neural_query_valid_seconds
neural_candidate_valid_seconds
neural_padding_seconds
neural_padding_ratio
neural_reliability
```

## Training Data

Each example is a pair of 5-second model inputs:

```text
query_audio_window
candidate_audio_window
label: 1 for same, 0 for not_same
metadata:
  query_song_id
  candidate_song_id
  query_start_seconds
  candidate_start_seconds
  pair_type
  query_valid_seconds
  padding_ratio
```

### Positive Pairs

Use a mix:

- 70% `positive_same_time`
- 30% `positive_jittered`

Jittered positives:

- Same song.
- Candidate start near query start.
- Jitter range around +/- 0.5 seconds.

The query side should receive realistic augmentations. The candidate side should usually remain cleaner because runtime candidates are loaded from stored database files.

Useful query augmentations:

- Noise.
- Volume changes.
- Light time stretch.
- Later: low-pass or compression-like perturbations if needed.

### Negative Pairs

Use a mix:

- Random negatives from different songs.
- Hard negatives from real fingerprint false positives.

Initial ratio:

- 70-80% random negatives.
- 20-30% hard negatives.

Later ratio after enough real failures are collected:

- 50% random negatives.
- 50% hard negatives.

Hard negative source:

- Test runner cases where `expected_song_id != selected_song_id` and `selected_song_id != -1`.
- Query comes from the expected song.
- Candidate fragment comes from the incorrectly selected song using the fingerprint-estimated offset.

### Query Duration Augmentation

Training must include variable query durations because runtime queries may be shorter than 5 seconds.

Suggested distribution:

- 70% query duration = 5.0 seconds.
- 20% query duration = 3.0 to 5.0 seconds, zero-padded.
- 10% query duration = 2.0 to 3.0 seconds, zero-padded.

Duration augmentation applies to positive and negative pairs. Reports must break quality down by duration bucket.

## Validation

Use two validation sets.

### `validation_known`

Uses songs that also exist in training, but with different windows, augmentations, and error cases.

Purpose:

- Estimate usefulness on the current fixed music database.

### `validation_heldout`

Uses songs completely excluded from training.

Purpose:

- Check whether the model learned a general comparison behavior instead of only memorizing songs.

## Metrics And Reports

Use B + C reporting: confusion matrices and pipeline-oriented reports.

### ML Evaluation Report

For each validation set and each threshold, report:

```text
TP
FP
FN
TN
precision
recall
false_positive_rate
false_negative_rate
```

Recommended thresholds:

- 0.50
- 0.70
- 0.85

Report slices:

- Pair type:
  - `positive_same_time`
  - `positive_jittered`
  - `negative_random`
  - `negative_hard`
- Query duration bucket:
  - `5.0s`
  - `4-5s`
  - `3-4s`
  - `2-3s`
- Validation type:
  - `known`
  - `heldout`

### Pipeline Shadow Report

Run from real test-runner searches where fingerprint results and neural observations are both available.

Report:

```text
total_searches
fingerprint_correct
fingerprint_wrong
fingerprint_not_found

neural_checked
neural_skipped
neural_errors

wrong fingerprint cases:
  neural_would_reject
  neural_would_confirm_wrong

weak correct cases:
  neural_would_confirm
  neural_would_reject

top-N cases:
  correct_song_present_in_candidates
  neural_highest_probability_is_correct
```

The report should help decide when shadow mode can be promoted to gray-zone validation.

## Trace Additions

`SearchTrace` should include:

```text
top_candidates
neural_enabled
neural_checked
neural_reason
neural_results
neural_error
```

Neural candidate result:

```text
song_id
rank
fingerprint_score
fingerprint_max_count
fingerprint_time_offset_seconds
same_probability
decision
threshold
reliability
query_valid_seconds
candidate_valid_seconds
padding_ratio
```

## Testing

Unit tests:

- Top candidate construction.
- Time offset conversion from bins to seconds.
- Query crop and zero-padding to 5 seconds.
- Candidate crop around offset.
- Reliability by query duration.
- Neural trace fields populated.
- Neural failure does not break search.

Offline tests:

- Generated dataset includes labels, pair types, duration buckets, and padding metadata.
- Evaluation report computes TP, FP, FN, and TN correctly.

Integration tests:

- Shadow mode preserves original `song_id` and `time_offset`.
- Neural results appear in `SearchTrace`.
- Missing model or PyTorch failure is recorded without failing recognition.

## Risks

1. Random negatives may be too easy.
   Hard negatives are required to target real false positives.

2. The `abs_diff` channel is sensitive to offset errors.
   Jittered positives are required. Later, the validator may evaluate multiple windows around the estimated offset.

3. Five seconds may be too short for repetitive or low-information sections.
   Shadow reports will reveal whether this is acceptable.

4. Zero padding can become a shortcut.
   Duration augmentation must apply to both positive and negative pairs, and reports must be bucketed by query duration.

5. Known-song validation can overstate quality.
   Heldout-song validation is required as a sanity check.

6. Neural dependencies must not destabilize search.
   Shadow mode errors are trace data, not recognition failures.

7. Loading full audio files for top-N candidates may be slow.
   This is acceptable for MVP, but later versions may need audio caching or precomputed features.

## Explicit Non-Goals For MVP

- Replace fingerprint search.
- Use neural embeddings for retrieval.
- Change returned search results.
- Build a vector database.
- Train two separate input-mode models.
- Implement gray-zone decision logic before shadow reports justify it.

## Open Decisions For Implementation Planning

These can be decided in the implementation plan:

- Exact log-mel parameters.
- Exact pair classifier CNN architecture.
- Config names and defaults.
- Top-N default value.
- Threshold defaults for reporting and shadow decisions.
- Storage format for hard negative examples.
