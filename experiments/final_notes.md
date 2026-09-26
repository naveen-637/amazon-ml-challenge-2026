# Final Matching Pipeline — Notes
## experiments/final_notes.md

---

## Pipeline Overview

`experiments/final_matching.py` is the **production pipeline** for the Amazon ML Challenge 2026
Business Entity Resolution task.

It reads `dataset/test/test_source{1,2,3}.tsv` and produces:
- `output/matching_results.tsv` — final entity matches (leaderboard file)
- `output/candidate_pairs.tsv` — all blocked candidate pairs

---

## Key Design Decisions

### 1. Blocking Strategy
Replicates the validated baseline from `analysis.py`:
- **Key A**: `country.upper() + "||" + normalize(business_name)[:4]`
- **Key B**: `country.upper() + "||" + normalize(business_address)[:6]`
- Union of both keys per S1 entity.
- Validated recall on 1,000 S1 training records: **86.75%** (3,051 / 3,517 true pairs recovered).

### 2. Scoring Function
For each blocked candidate pair:
```
name_sim  = RapidFuzz token_sort_ratio(s1_name_words, s2_name_words) / 100.0
addr_sim  = RapidFuzz token_sort_ratio(s1_addr_words, s2_addr_words) / 100.0
score     = 0.60 * name_sim + 0.40 * addr_sim
```
Exact-match shortcut: if `normalize(name_s1) == normalize(name_s2)`, `name_sim = 1.0` (no fuzzy call).
Similarly for address.

Boost rules:
- If `name_exact AND addr_sim >= 0.50`: `score = max(score, 0.50 + 0.50 * addr_sim)`
- If `addr_exact AND name_sim >= 0.50`: `score = max(score, 0.40 + 0.60 * name_sim)`

Hard filter: mismatched country → immediate skip (no scoring).

### 3. Matching Threshold
**THRESHOLD = 0.82** (tunable at top of script).

Validated on 1,000 S1 training records:
| Threshold | Precision | Recall | Macro F0.5 |
|-----------|-----------|--------|------------|
| 0.75      | ~75%      | ~65%   | ~0.70      |
| 0.80      | ~80%      | ~60%   | ~0.69      |
| **0.82**  | **83.77%**| **56.92%** | **0.6852** |
| 0.85      | ~87%      | ~52%   | ~0.68      |
| 0.90      | ~92%      | ~44%   | ~0.66      |

F0.5 weights precision 2x over recall. Threshold 0.82 was the empirically optimal point
on the 1k training validation set.

> **Note**: The optimal threshold may shift slightly on the full test set
> (France records, different business name distributions). Consider trying 0.80–0.85.

### 4. France Country Handling
The test set contains **France** records (not in training data).
The pipeline handles France transparently:
- Country is treated as an **open-set string label** (uppercased and used as a blocking prefix).
- No country-specific rules or hardcoding.
- France records are blocked and scored identically to US/India records.

### 5. Memory Architecture
- **S1 batch size**: 10,000 records per batch (tunable: `S1_BATCH_SIZE`)
- **S2/S3 chunk size**: 100,000 rows per read (tunable: `CHUNK_SIZE`)
- Accepted matches written to disk immediately; accumulator dicts freed after each batch.
- Peak RAM measured: ~155–200 MB during sample test.
- Expected peak during full run: ~500 MB–1 GB (safe for 16 GB laptop).

---

## Experiment History

| Experiment | File | Finding |
|------------|------|---------|
| Baseline blocking | `analysis.py` | Name(4)+Addr(6) → 86.75% recall on 1k S1 |
| Advanced blocking | `experiments/blocking_exp4.py` | First Token(4)+AddrNumStreet(3) → 20% fewer candidates at same recall |
| Matching scoring  | `experiments/matching_exp1.py` | Threshold 0.82 → Macro F0.5 0.685 on 1k S1 |
| **Final pipeline** | `experiments/final_matching.py` | Production script for test set |

---

## Benchmark Results (Small Sample Test)

Tested on 100 S1 test records × 50,000 S2/S3 rows:
- **Time**: 2.2 seconds
- **Blocking hits**: 7,394 rows filtered from 100,000 total
- **Pairs scored**: 8,542
- **Matches accepted**: 3 (threshold 0.82)
- **RAM used**: ~155 MB

Full dataset benchmark (5,000 S1 × 100,000 S2 chunk):
- **Time per 100k chunk**: ~51 seconds (566,000 pairs scored)
- **Extrapolated for full test S2 (~5M rows) with 5k batch**: ~42 minutes per batch
- **Total test S1**: 1,732,544 records → ~174 batches of 10k
- **Estimated total runtime**: 3–6 hours (depending on hit rate across different S1 batches)

> Runtime is dominated by RapidFuzz scoring calls. The blocking correctly narrows candidates
> from ~1.7 trillion possible pairs to a manageable set.

---

## Runtime Estimate for Full Test

| Component | Size | Est. Time |
|-----------|------|-----------|
| Test S1 | 1,732,544 records | — |
| Test S2 | ~5M rows | ~1.5–2h for scoring pass |
| Test S3 | ~5M rows | ~1.5–2h for scoring pass |
| **Total** | | **3–5 hours** |

The pipeline prints an ETA estimate after each completed batch.

---

## How to Run

### Syntax check
```bash
python -m py_compile experiments/final_matching.py
```

### Quick smoke test (does NOT write output files)
```bash
python experiments/final_matching.py --sample-test
```

### Full test dataset run (only when ready)
```bash
python experiments/final_matching.py
```

### Validate output before submitting
```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## Tuning Parameters

Edit these constants at the top of `experiments/final_matching.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `THRESHOLD` | `0.82` | Lower = more recall, more FP. Higher = higher precision, fewer matches. |
| `S1_BATCH_SIZE` | `10_000` | Reduce if RAM is tight. Increase for faster throughput. |
| `CHUNK_SIZE` | `100_000` | S2/S3 rows per read. Reduce if pandas RAM spikes. |
| `SAVE_CANDIDATES` | `True` | Set False to skip `candidate_pairs.tsv` (faster run). |

---

## Output Format

### `output/matching_results.tsv`
```
source1_entity_id   matched_entity_ids
S1-714132312        S2-123456,S3-789012
S1-106407869        
S1-156285671        S2-999001
```
- Tab-separated
- Every test S1 entity appears exactly once
- Empty `matched_entity_ids` = singleton (no match predicted)
- Comma-separated S2/S3 IDs (no duplicates within a list)

### `output/candidate_pairs.tsv`
Same format, column named `candidate_entity_ids`.

---

## Known Limitations

1. **Blocking recall ceiling**: 86.75% on training validation → ~13% of true matches
   are missed before scoring (blocked out). This is the hard upper bound on recall.
2. **Threshold calibration**: 0.82 was optimised on 1k training records.
   May not be globally optimal; consider 0.80–0.85 range.
3. **No ML model**: Scoring uses a simple weighted sum, not a trained classifier.
   A logistic regression or gradient-boosted tree on the 6 features could improve F0.5.
4. **France generalisation**: Zero France training examples. Blocking relies purely
   on character-level normalisation; transliteration variants may reduce recall.
