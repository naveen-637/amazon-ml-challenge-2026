"""
Matching Experiment 1: Memory-Efficient Candidate Scoring & Threshold Evaluation
================================================================================
Takes the baseline blocking approach from analysis.py:
  - Country + normalized business name first 4 characters
  - Country + normalized business address first 6 characters
  - Union of these candidate pairs

Builds a memory-efficient candidate scoring pipeline on a validation sample of
1,000 S1 records against S2 and S3 (processed in manageable chunks).

Similarity Features Calculated per Candidate Pair:
  1. Normalized business-name exact match (0.0 or 1.0)
  2. Normalized business-name RapidFuzz similarity (0.0 to 1.0)
  3. Normalized address exact match (0.0 or 1.0)
  4. Normalized address RapidFuzz similarity (0.0 to 1.0)
  5. Country exact match (0.0 or 1.0)
  6. Combined weighted name/address similarity score (0.0 to 1.0)

Evaluates multiple conservative matching thresholds against Ground Truth
using the official competition metric:
  F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
reporting both Pair-level and Macro-averaged (per S1 entity) metrics,
along with predicted singleton counts.

Memory & Runtime Architecture:
  - Streaming chunked processing of S2 & S3 (100,000 rows/chunk).
  - Pre-filtered vectorized candidate masks via .isin() in C.
  - On-the-fly candidate pair scoring without materializing 10M+ pairs into memory.
  - Discards low-score pairs (<0.55) immediately to maintain RAM below 70-75%.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

# ----------------------------------------------------------------------
# 0. DATASET DISCOVERY
# ----------------------------------------------------------------------
possible_bases = [
    Path("dataset/train"),
    Path("student_resource/dataset/train"),
    Path(__file__).resolve().parent.parent / "dataset" / "train",
    Path(__file__).resolve().parent / "dataset" / "train",
    Path(__file__).resolve().parent.parent / "student_resource" / "dataset" / "train",
]
BASE = None
for p in possible_bases:
    if p.exists():
        BASE = str(p)
        break

if BASE is None:
    BASE = "dataset/train"

print(f"Dataset directory: {BASE}")

cols = ["entity_id", "business_name", "business_address", "country"]

# Evaluation thresholds to test (conservative focus for F0.5 optimization)
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
MIN_SCORE_CUTOFF = min(THRESHOLDS)


# ----------------------------------------------------------------------
# 1. TEXT NORMALIZATION HELPER FUNCTIONS
# ----------------------------------------------------------------------
def normalize_text_clean(series: pd.Series) -> pd.Series:
    """Lowercase alphanumeric string (no spaces/punctuation) for exact match & blocking."""
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def normalize_text_words(series: pd.Series) -> pd.Series:
    """Lowercase string with normalized whitespace for token-based fuzzy scoring."""
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ----------------------------------------------------------------------
# 2. LOAD S1 VALIDATION SAMPLE (1,000 RECORDS) & BUILD BLOCKING INDEX
# ----------------------------------------------------------------------
print("=" * 85)
print("MATCHING EXPERIMENT 1: CANDIDATE SCORING & F0.5 THRESHOLD CALIBRATION")
print("=" * 85)
print("Loading 1,000 S1 records for validation experiment...")

s1 = pd.read_csv(f"{BASE}/train_source1.tsv", sep="\t", usecols=cols)
sample_s1 = s1.head(1000).copy()
sample_ids = list(sample_s1["entity_id"])
sample_id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

# Feature extraction for S1
s1_country = sample_s1["country"].fillna("").astype(str).str.strip().str.upper().values
s1_name_clean = normalize_text_clean(sample_s1["business_name"]).values
s1_addr_clean = normalize_text_clean(sample_s1["business_address"]).values
s1_name_words = normalize_text_words(sample_s1["business_name"]).values
s1_addr_words = normalize_text_words(sample_s1["business_address"]).values

# Build baseline blocking lookups from analysis.py:
# 1. Country + normalized name first 4 characters
# 2. Country + normalized address first 6 characters
lookup_name4 = {}
lookup_addr6 = {}

for idx in range(len(sample_s1)):
    c = s1_country[idx]
    nc = s1_name_clean[idx]
    ac = s1_addr_clean[idx]

    kn = c + "||" + nc[:4]
    ka = c + "||" + ac[:6]

    if kn and not kn.endswith("||"):
        lookup_name4.setdefault(kn, []).append(idx)
    if ka and not ka.endswith("||"):
        lookup_addr6.setdefault(ka, []).append(idx)

valid_name4 = set(lookup_name4.keys())
valid_addr6 = set(lookup_addr6.keys())

print(f"S1 Name Norm(4) blocking keys  : {len(valid_name4):,}")
print(f"S1 Addr Norm(6) blocking keys  : {len(valid_addr6):,}")


# ----------------------------------------------------------------------
# 3. LOAD GROUND TRUTH LABELS FOR VALIDATION SAMPLE
# ----------------------------------------------------------------------
print("\nLoading ground truth matches for S1 sample...")
gt_pairs = set()  # set of (s1_id, target_id)
gt_matches_per_s1 = {sid: set() for sid in sample_ids}

for chunk in pd.read_csv(
    f"{BASE}/train_ground_truth.tsv",
    sep="\t",
    chunksize=250_000
):
    matched_gt = chunk[chunk["source1_entity_id"].isin(sample_id_to_idx)]
    for row in matched_gt.itertuples(index=False):
        matched = getattr(row, "matched_entity_ids", "")
        if pd.isna(matched) or not matched:
            continue
        s1_id = getattr(row, "source1_entity_id")
        for tid in str(matched).split(","):
            tid = tid.strip()
            if tid:
                gt_pairs.add((s1_id, tid))
                gt_matches_per_s1[s1_id].add(tid)

total_gt_pairs = len(gt_pairs)
gt_singletons_count = sum(1 for sid, targets in gt_matches_per_s1.items() if len(targets) == 0)

print(f"Total ground-truth pairs to recover : {total_gt_pairs:,}")
print(f"Ground-truth singletons in S1 sample: {gt_singletons_count:,} / {len(sample_s1):,}")


# ----------------------------------------------------------------------
# 4. STREAMING CHUNKED CANDIDATE GENERATION & IN-LINE SCORING
# ----------------------------------------------------------------------
# Predicted pairs per threshold: threshold -> set of (s1_id, target_id)
pred_pairs = {t: set() for t in THRESHOLDS}
# S1 entities with at least one prediction: threshold -> set of s1_ids
s1_with_preds = {t: set() for t in THRESHOLDS}
# Per-entity predictions: threshold -> dict of s1_id -> set of predicted target_ids
preds_by_s1 = {t: {sid: set() for sid in sample_ids} for t in THRESHOLDS}

source_files = [
    (f"{BASE}/train_source2.tsv", "Source 2"),
    (f"{BASE}/train_source3.tsv", "Source 3"),
]

total_candidates_blocked = 0
total_candidates_scored = 0
total_s2_s3_rows = 0
chunk_size = 100_000

print("\n" + "=" * 85)
print("STREAMING CHUNK PROCESSING & CANDIDATE SCORING (LOW MEMORY)")
print("=" * 85)
start_time = time.time()

for file_path, source_name in source_files:
    print(f"\nProcessing {source_name} ({file_path}) in chunks of {chunk_size:,}...")
    chunk_reader = pd.read_csv(
        file_path,
        sep="\t",
        usecols=cols,
        chunksize=chunk_size
    )

    for chunk_num, chunk in enumerate(chunk_reader, start=1):
        chunk_len = len(chunk)
        total_s2_s3_rows += chunk_len

        # Vectorized feature generation for chunk
        c_country = chunk["country"].fillna("").astype(str).str.strip().str.upper()
        c_name_clean = normalize_text_clean(chunk["business_name"])
        c_addr_clean = normalize_text_clean(chunk["business_address"])
        c_name_words = normalize_text_words(chunk["business_name"])
        c_addr_words = normalize_text_words(chunk["business_address"])
        c_entity_ids = chunk["entity_id"].values

        # Vectorized blocking keys
        k_name = c_country + "||" + c_name_clean.str[:4]
        k_addr = c_country + "||" + c_addr_clean.str[:6]

        # Vectorized candidate selection via C hashtable lookup
        mask_name = k_name.isin(valid_name4)
        mask_addr = k_addr.isin(valid_addr6)
        mask_any = mask_name | mask_addr

        matched_indices = np.where(mask_any)[0]
        if len(matched_indices) == 0:
            continue

        # Extract values for fast array indexing (no iterrows)
        sub_ids = c_entity_ids[matched_indices]
        sub_name_clean = c_name_clean.values[matched_indices]
        sub_addr_clean = c_addr_clean.values[matched_indices]
        sub_name_words = c_name_words.values[matched_indices]
        sub_addr_words = c_addr_words.values[matched_indices]
        sub_country = c_country.values[matched_indices]
        sub_k_name = k_name.values[matched_indices]
        sub_k_addr = k_addr.values[matched_indices]
        sub_mask_name = mask_name.values[matched_indices]
        sub_mask_addr = mask_addr.values[matched_indices]

        # On-the-fly candidate scoring
        for row_i in range(len(matched_indices)):
            target_id = sub_ids[row_i]
            t_country = sub_country[row_i]
            t_name_c = sub_name_clean[row_i]
            t_addr_c = sub_addr_clean[row_i]
            t_name_w = sub_name_words[row_i]
            t_addr_w = sub_addr_words[row_i]

            # Collect unique S1 indices for this target row (union of blocking keys)
            s1_cands = set()
            if sub_mask_name[row_i]:
                for s1_i in lookup_name4[sub_k_name[row_i]]:
                    s1_cands.add(s1_i)
            if sub_mask_addr[row_i]:
                for s1_i in lookup_addr6[sub_k_addr[row_i]]:
                    s1_cands.add(s1_i)

            total_candidates_blocked += len(s1_cands)

            # Score each candidate pair
            for s1_i in s1_cands:
                total_candidates_scored += 1
                s1_id = sample_ids[s1_i]

                # 5. Country exact match
                country_exact = 1.0 if s1_country[s1_i] == t_country else 0.0
                if country_exact == 0.0:
                    continue  # Mismatched country cannot be a valid match

                # 1. Normalized business-name exact match
                name_exact = 1.0 if s1_name_clean[s1_i] == t_name_c else 0.0

                # 3. Normalized address exact match
                addr_exact = 1.0 if s1_addr_clean[s1_i] == t_addr_c else 0.0

                # 2. Normalized business-name RapidFuzz similarity
                if name_exact == 1.0:
                    name_sim = 1.0
                else:
                    name_sim = fuzz.token_sort_ratio(s1_name_words[s1_i], t_name_w) / 100.0

                # 4. Normalized address RapidFuzz similarity
                if addr_exact == 1.0:
                    addr_sim = 1.0
                else:
                    addr_sim = fuzz.token_sort_ratio(s1_addr_words[s1_i], t_addr_w) / 100.0

                # 6. Combined name/address similarity score
                # Base weighting: 60% business name, 40% address
                combined_score = 0.60 * name_sim + 0.40 * addr_sim

                # Boost if exact match present on one signal
                if name_exact == 1.0 and addr_sim >= 0.50:
                    combined_score = max(combined_score, 0.50 + 0.50 * addr_sim)
                elif addr_exact == 1.0 and name_sim >= 0.50:
                    combined_score = max(combined_score, 0.40 + 0.60 * name_sim)

                # Only retain pairs scoring above the minimum threshold
                if combined_score < MIN_SCORE_CUTOFF:
                    continue

                # Record predictions across qualifying thresholds
                pair = (s1_id, target_id)
                for t in THRESHOLDS:
                    if combined_score >= t:
                        pred_pairs[t].add(pair)
                        s1_with_preds[t].add(s1_id)
                        preds_by_s1[t][s1_id].add(target_id)

        print(f"  -> {source_name} Chunk {chunk_num:2d}: {chunk_len:,} rows | Scored candidates: {total_candidates_scored:,}", end="\r")

    print()  # Advance past carriage return

elapsed = time.time() - start_time
print(f"\nScoring completed in {elapsed:.2f} seconds.")
print(f"Total S2 + S3 records processed: {total_s2_s3_rows:,}")
print(f"Total candidate pairs generated: {total_candidates_blocked:,}")


# ----------------------------------------------------------------------
# 5. METRIC EVALUATION ACROSS CONSERVATIVE THRESHOLDS
# ----------------------------------------------------------------------
print("\n" + "=" * 115)
print("MATCHING EXPERIMENT 1: THRESHOLD EVALUATION REPORT (F0.5 METRIC)")
print("=" * 115)
print(f"S1 Validation Sample Size       : {len(sample_s1):,}")
print(f"Total Ground-Truth Pairs        : {total_gt_pairs:,}")
print(f"Total Ground-Truth Singletons   : {gt_singletons_count:,}")
print(f"Evaluation Metric Focus         : F0.5 (Precision Weighted 2x over Recall)")
print("-" * 115)

header = (
    f"{'Threshold':<10} | "
    f"{'Pred.Pairs':<11} | "
    f"{'TP':<7} | "
    f"{'FP':<8} | "
    f"{'FN':<7} | "
    f"{'Precision':<10} | "
    f"{'Recall':<8} | "
    f"{'Pair F0.5':<10} | "
    f"{'Macro F0.5':<11} | "
    f"{'Pred.Singletons'}"
)
print(header)
print("-" * len(header))

best_thresh = None
best_macro_f05 = -1.0
best_stats = {}

for t in THRESHOLDS:
    preds = pred_pairs[t]
    n_pred = len(preds)

    # Pair-level confusion matrix
    tp = len(preds & gt_pairs)
    fp = n_pred - tp
    fn = total_gt_pairs - tp

    precision = (tp / n_pred) if n_pred > 0 else 1.0
    recall = (tp / total_gt_pairs) if total_gt_pairs > 0 else 0.0

    # Pair-level F0.5
    denom = 0.25 * precision + recall
    pair_f05 = (1.25 * precision * recall / denom) if denom > 0 else 0.0

    # Official Macro-averaged F0.5 (per S1 entity as defined in challenge README)
    entity_f05_scores = []
    for s1_id in sample_ids:
        true_targets = gt_matches_per_s1[s1_id]
        pred_targets = preds_by_s1[t][s1_id]

        if len(true_targets) == 0:
            # Singleton entity: 1.0 if empty prediction, 0.0 otherwise
            score = 1.0 if len(pred_targets) == 0 else 0.0
        else:
            # Entity with true matches
            if len(pred_targets) == 0:
                score = 0.0
            else:
                e_tp = len(pred_targets & true_targets)
                if e_tp == 0:
                    score = 0.0
                else:
                    e_prec = e_tp / len(pred_targets)
                    e_rec = e_tp / len(true_targets)
                    e_denom = 0.25 * e_prec + e_rec
                    score = (1.25 * e_prec * e_rec / e_denom) if e_denom > 0 else 0.0

        entity_f05_scores.append(score)

    macro_f05 = np.mean(entity_f05_scores)
    pred_singletons = len(sample_s1) - len(s1_with_preds[t])

    if macro_f05 > best_macro_f05:
        best_macro_f05 = macro_f05
        best_thresh = t
        best_stats = {
            "threshold": t,
            "pred_pairs": n_pred,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "pair_f05": pair_f05,
            "macro_f05": macro_f05,
            "singletons": pred_singletons,
        }

    print(
        f"{t:<10.2f} | "
        f"{n_pred:<11,d} | "
        f"{tp:<7,d} | "
        f"{fp:<8,d} | "
        f"{fn:<7,d} | "
        f"{precision:<10.2%} | "
        f"{recall:<8.2%} | "
        f"{pair_f05:<10.4f} | "
        f"{macro_f05:<11.4f} | "
        f"{pred_singletons:<15,d}"
    )

print("=" * 115)
print("\nOPTIMAL CONSERVATIVE THRESHOLD SUMMARY:")
print(f"  * Best Threshold (Macro F0.5) : {best_thresh:.2f}")
print(f"  * Macro F0.5 Score            : {best_stats['macro_f05']:.4f}")
print(f"  * Pair-Level F0.5 Score       : {best_stats['pair_f05']:.4f}")
print(f"  * Precision                   : {best_stats['precision']:.2%}")
print(f"  * Recall                      : {best_stats['recall']:.2%}")
print(f"  * Predicted Matches           : {best_stats['pred_pairs']:,} (TP: {best_stats['tp']:,}, FP: {best_stats['fp']:,})")
print(f"  * Predicted Singletons        : {best_stats['singletons']:,} (GT Singletons: {gt_singletons_count:,})")
print("=" * 115)
print("\nExperiment matching_exp1.py execution completed successfully.")
