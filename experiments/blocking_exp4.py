"""
Blocking Experiment 4: High-Recall, Selective Blocking Strategies
================================================================
Investigates selective blocking keys using normalized business names and addresses
to improve blocking recall while substantially reducing candidate counts.

Compared against baseline:
  Country + Name Norm (4 chars) + Country + Address Norm (6 chars)
  [Baseline Recall: 86.75%, Candidates: 10,308,059]

Strategies Evaluated:
1. First character / first 2 characters of normalized name combined with country.
2. Last 4 characters of normalized name combined with country (raw vs suffix-stripped).
3. Name token-based blocking (first token, longest distinctive token).
4. Address number + location/token based blocking (house number + street prefix).
5. Compound name and address signals (Name(3)+AddrNum, Name(3)+Addr(3), Token(4)+AddrNum).
6. Multi-key ensemble combinations designed for high recall and minimal candidate volume.

Runtime & Memory Optimizations:
- Chunked processing of S2 and S3 (250,000 rows/chunk).
- Vectorized pandas / regex feature extraction (no iterrows).
- Fast C-hash candidate filtering via .isin(s1_valid_keys).
- Memory-efficient 64-bit packed candidate representation:
    (s1_index << 33) | (source_flag << 32) | target_id_number
  enabling safe execution within standard RAM limits without OOM.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# 0. LOCATE DATASET DIRECTORY
# ----------------------------------------------------------------------
possible_bases = [
    Path("dataset/train"),
    Path("student_resource/dataset/train"),
    Path(__file__).resolve().parent.parent / "dataset" / "train",
    Path(__file__).resolve().parent / "dataset" / "train",
]
BASE = None
for p in possible_bases:
    if p.exists():
        BASE = str(p)
        break

if BASE is None:
    BASE = "dataset/train"

print(f"Using dataset base directory: {BASE}")

cols = ["entity_id", "business_name", "business_address", "country"]

# Common corporate legal suffixes to strip for distinctive token/suffix extraction
COMMON_LEGAL_TOKENS = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "co", "company", "pvt", "private", "pc", "services", "group", "holdings",
    "enterprises", "llp", "gmbh", "sa", "ag", "srl", "bv"
}

LEGAL_REGEX_PATTERN = r"\b(" + "|".join(COMMON_LEGAL_TOKENS) + r")\b"


# ----------------------------------------------------------------------
# 1. VECTORIZED FEATURE EXTRACTION FUNCTIONS
# ----------------------------------------------------------------------
def extract_blocking_features(df: pd.DataFrame) -> dict:
    """
    Vectorized extraction of all blocking signals for a DataFrame chunk.
    Returns a dictionary of key series corresponding to candidate blocking keys.
    """
    # Country: cleaned uppercase string
    country = df["country"].fillna("").astype(str).str.strip().str.upper()

    # 1. Normalized business name (alphanumeric only, no spaces)
    name_clean = (
        df["business_name"].fillna("").astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

    # 2. Tokenized business name (alphanumeric with spaces)
    name_words = (
        df["business_name"].fillna("").astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.strip()
    )

    # First word token from business name
    name_first_token = name_words.str.split().str[0].fillna("")

    # Longest distinctive token (filtered for common legal suffixes if length >= 4)
    def _get_distinctive_token(text: str) -> str:
        tokens = text.split()
        if not tokens:
            return ""
        filtered = [t for t in tokens if t not in COMMON_LEGAL_TOKENS and len(t) >= 3]
        if filtered:
            return max(filtered, key=len)
        return max(tokens, key=len)

    name_longest_token = name_words.map(_get_distinctive_token)

    # Suffix-cleaned name: remove common corporate entity suffixes, then collapse
    name_stripped = (
        name_words
        .str.replace(LEGAL_REGEX_PATTERN, "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )

    # 3. Normalized address (alphanumeric only, no spaces)
    addr_clean = (
        df["business_address"].fillna("").astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

    # Address raw string
    addr_raw = df["business_address"].fillna("").astype(str).str.strip()

    # Address House/Building Number: first sequence of 1 to 6 digits
    addr_number = addr_raw.str.extract(r"(\b\d{1,6}\b)")[0].fillna("")

    # First alphabetical word of address (street name or city)
    addr_words = (
        addr_raw.str.lower()
        .str.replace(r"[^a-z\s]", " ", regex=True)
        .str.strip()
    )
    first_street_word = addr_words.str.split().str[0].fillna("")

    # ------------------------------------------------------------------
    # KEY DICTIONARY GENERATION
    # ------------------------------------------------------------------
    keys = {}

    # Baseline keys (as in analysis.py)
    keys["base_name4"] = country + "||" + name_clean.str[:4]
    keys["base_addr6"] = country + "||" + addr_clean.str[:6]

    # Investigation 1: First character / first 2 characters + country
    keys["strat_name_char1"] = country + "||" + name_clean.str[:1]
    keys["strat_name_char2"] = country + "||" + name_clean.str[:2]

    # Investigation 2: Last 4 characters + country (raw suffix vs legal-cleaned suffix)
    keys["strat_name_last4"] = country + "||" + name_clean.str[-4:]
    keys["strat_name_clean_last4"] = country + "||" + name_stripped.str[-4:]

    # Investigation 3: Name token based blocking
    keys["strat_name_first_token4"] = country + "||" + name_first_token.str[:4]
    keys["strat_name_longest_token5"] = country + "||" + name_longest_token.str[:5]

    # Investigation 4: Address number + location/street token
    keys["strat_addr_num_street3"] = country + "||" + addr_number + "||" + first_street_word.str[:3]
    keys["strat_addr_num_only"] = country + "||" + addr_number

    # Investigation 5: Selective compound name + address signals
    # 5a: Country + Name Norm (3 chars) + Address Number
    keys["strat_compound_name3_addr_num"] = country + "||" + name_clean.str[:3] + "||" + addr_number

    # 5b: Country + Name Norm (3 chars) + Address Norm (3 chars)
    keys["strat_compound_name3_addr3"] = country + "||" + name_clean.str[:3] + "||" + addr_clean.str[:3]

    # 5c: Country + Distinctive Token (4 chars) + Address Number
    keys["strat_compound_token4_addr_num"] = country + "||" + name_longest_token.str[:4] + "||" + addr_number

    return keys


# ----------------------------------------------------------------------
# 2. LOAD S1 (1,000 SAMPLE) AND PREPARE LOOKUPS
# ----------------------------------------------------------------------
print("=" * 80)
print("BLOCKING EXPERIMENT 4: SELECTIVE BLOCKING INVESTIGATION")
print("=" * 80)
print("Loading first 1,000 S1 records...")

s1 = pd.read_csv(f"{BASE}/train_source1.tsv", sep="\t", usecols=cols)
sample_s1 = s1.head(1000).copy()
sample_ids = list(sample_s1["entity_id"])
sample_id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

print("Computing S1 blocking keys...")
s1_keys = extract_blocking_features(sample_s1)

# Strategy metadata registry
STRATEGY_METADATA = [
    # Baseline
    {
        "id": "base_name4",
        "category": "Baseline",
        "name": "Country + Name Norm (4 chars)",
        "desc": "Baseline name prefix key",
    },
    {
        "id": "base_addr6",
        "category": "Baseline",
        "name": "Country + Address Norm (6 chars)",
        "desc": "Baseline address prefix key",
    },
    # Inv 1: First 1 & 2 characters
    {
        "id": "strat_name_char1",
        "category": "1. Char Prefixes",
        "name": "Country + Name Norm (1 char)",
        "desc": "First character of normalized name + country",
    },
    {
        "id": "strat_name_char2",
        "category": "1. Char Prefixes",
        "name": "Country + Name Norm (2 chars)",
        "desc": "First 2 characters of normalized name + country",
    },
    # Inv 2: Last 4 characters
    {
        "id": "strat_name_last4",
        "category": "2. Name Suffixes",
        "name": "Country + Name Norm (Last 4 chars)",
        "desc": "Last 4 characters of raw normalized name + country",
    },
    {
        "id": "strat_name_clean_last4",
        "category": "2. Name Suffixes",
        "name": "Country + Name Clean-Suffix (Last 4)",
        "desc": "Last 4 characters after stripping legal corporate terms",
    },
    # Inv 3: Name token based
    {
        "id": "strat_name_first_token4",
        "category": "3. Name Tokens",
        "name": "Country + Name First Token (4 chars)",
        "desc": "First word token of business name + country",
    },
    {
        "id": "strat_name_longest_token5",
        "category": "3. Name Tokens",
        "name": "Country + Name Longest Token (5 chars)",
        "desc": "Longest distinctive word token (>=3 chars, non-legal)",
    },
    # Inv 4: Address number + location/token
    {
        "id": "strat_addr_num_street3",
        "category": "4. Address Tokens",
        "name": "Country + Addr Num + Street Token (3)",
        "desc": "House/street number + first 3 letters of street name",
    },
    {
        "id": "strat_addr_num_only",
        "category": "4. Address Tokens",
        "name": "Country + Addr Number Alone",
        "desc": "Country + first street/building number",
    },
    # Inv 5: Compound name + address signals
    {
        "id": "strat_compound_name3_addr_num",
        "category": "5. Compound Keys",
        "name": "Country + Name (3) + Addr Number",
        "desc": "Compound: first 3 name chars + street number",
    },
    {
        "id": "strat_compound_name3_addr3",
        "category": "5. Compound Keys",
        "name": "Country + Name (3) + Addr Norm (3)",
        "desc": "Compound: first 3 name chars + first 3 address chars",
    },
    {
        "id": "strat_compound_token4_addr_num",
        "category": "5. Compound Keys",
        "name": "Country + Distinctive Token (4) + Addr Num",
        "desc": "Compound: first 4 distinctive token chars + street number",
    },
]

# Build S1 lookup dictionaries: key -> list of s1_indices
s1_lookups = {}
s1_valid_keys = {}

for strat in STRATEGY_METADATA:
    strat_id = strat["id"]
    series = s1_keys[strat_id]
    lookup = {}
    for s1_idx, val in enumerate(series):
        # Ignore empty values or empty key suffixes
        if val and not val.endswith("||"):
            lookup.setdefault(val, []).append(s1_idx)
    s1_lookups[strat_id] = lookup
    s1_valid_keys[strat_id] = set(lookup.keys())


# ----------------------------------------------------------------------
# 3. LOAD GROUND TRUTH (PRE-PACKED AS 64-BIT INTEGERS)
# ----------------------------------------------------------------------
print("Loading ground truth for 1,000 S1 records in chunks...")
true_pairs_set = set()

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
        s1_idx = sample_id_to_idx[s1_id]
        for tid in str(matched).split(","):
            tid = tid.strip()
            if tid:
                # Packed int: (s1_idx << 33) | (src_bit << 32) | id_number
                src_bit = 1 if tid.startswith("S3-") else 0
                num = int(tid[3:])
                true_pairs_set.add((s1_idx << 33) | (src_bit << 32) | num)

total_true = len(true_pairs_set)
print(f"Ground-truth pairs to recover for S1 sample: {total_true:,}")


# ----------------------------------------------------------------------
# 4. CHUNKED BLOCKING OVER S2 AND S3 (VECTORIZED & LOW-MEMORY)
# ----------------------------------------------------------------------
candidate_sets = {strat["id"]: set() for strat in STRATEGY_METADATA}

source_files = [
    (f"{BASE}/train_source2.tsv", 0, "Source 2"),
    (f"{BASE}/train_source3.tsv", 1, "Source 3"),
]

s2_count = 0
s3_count = 0
start_time = time.time()

print("\n" + "=" * 80)
print("PROCESSING S2 AND S3 IN CHUNKS (VECTORIZED FILTERING)")
print("=" * 80)

for file_path, src_bit, source_name in source_files:
    print(f"\nProcessing {source_name} ({file_path}) in chunks of 250,000...")
    chunk_reader = pd.read_csv(
        file_path,
        sep="\t",
        usecols=cols,
        chunksize=250_000
    )

    for chunk_number, chunk in enumerate(chunk_reader, start=1):
        chunk_len = len(chunk)
        if src_bit == 0:
            s2_count += chunk_len
        else:
            s3_count += chunk_len

        print(f"  -> {source_name} Chunk {chunk_number:2d}: {chunk_len:,} rows processed", end="\r")

        # 1. Vectorized key generation for chunk
        chunk_keys = extract_blocking_features(chunk)

        # 2. Extract numeric entity id for fast integer packing
        # Entity IDs have format 'S2-12345678' -> integer 12345678
        entity_num_ids = chunk["entity_id"].str[3:].astype(int).values

        # 3. Vectorized candidate matching for each strategy
        for strat in STRATEGY_METADATA:
            strat_id = strat["id"]
            valid_keys = s1_valid_keys[strat_id]
            lookup = s1_lookups[strat_id]

            key_series = chunk_keys[strat_id]
            mask = key_series.isin(valid_keys)

            if mask.any():
                matched_num_ids = entity_num_ids[mask]
                matched_keys = key_series[mask].values
                target_set = candidate_sets[strat_id]

                for tid, k in zip(matched_num_ids, matched_keys):
                    target_s1_indices = lookup[k]
                    for s1_idx in target_s1_indices:
                        # 64-bit packed pair representation
                        target_set.add((s1_idx << 33) | (src_bit << 32) | tid)

    print()  # Advance newline after carriage return

elapsed = time.time() - start_time
total_universe = s2_count + s3_count
possible_pairs = len(sample_s1) * total_universe

print(f"\nChunk processing completed in {elapsed:.2f} seconds.")
print(f"Total S2 records: {s2_count:,} | Total S3 records: {s3_count:,}")
print(f"Total candidate space (Cartesian product): {possible_pairs:,}")


# ----------------------------------------------------------------------
# 5. MULTI-KEY ENSEMBLE COMBINATIONS
# ----------------------------------------------------------------------
# 1. Baseline combination (from analysis.py)
baseline_set = candidate_sets["base_name4"].union(candidate_sets["base_addr6"])
baseline_cand_count = len(baseline_set)
baseline_found = len(true_pairs_set.intersection(baseline_set))
baseline_recall = (baseline_found / total_true) if total_true else 0.0

# 2. Combo A: Balanced High Efficiency (First Token 4 + Addr Number & Street 3)
combo_balanced = candidate_sets["strat_name_first_token4"].union(
    candidate_sets["strat_addr_num_street3"]
)

# 3. Combo B: High Recall Multi-Token (First Token 4 + Addr Num & Street 3 + Longest Distinctive Token 5)
combo_high_recall = (
    candidate_sets["strat_name_first_token4"]
    .union(candidate_sets["strat_addr_num_street3"])
    .union(candidate_sets["strat_name_longest_token5"])
)

# 4. Combo C: Ultra-Selective Compound Signals (Name3+AddrNum + Name3+Addr3 + Token4+AddrNum)
combo_ultra_selective = (
    candidate_sets["strat_compound_name3_addr_num"]
    .union(candidate_sets["strat_compound_name3_addr3"])
    .union(candidate_sets["strat_compound_token4_addr_num"])
)

# 5. Combo D: Name Norm(4) + Addr Number & Street(3) (Direct drop-in replacement for Baseline's Address key)
combo_name4_addr_street = candidate_sets["base_name4"].union(
    candidate_sets["strat_addr_num_street3"]
)

COMBINED_STRATEGIES = [
    {
        "id": "combo_baseline",
        "name": "BASELINE: Name Norm(4) + Addr Norm(6)",
        "set": baseline_set,
        "is_baseline": True,
    },
    {
        "id": "combo_balanced",
        "name": "COMBO: First Token(4) + Addr Num & Street(3)",
        "set": combo_balanced,
        "is_baseline": False,
    },
    {
        "id": "combo_high_recall",
        "name": "COMBO: First Token(4) + Addr Num & Street(3) + Longest Token(5)",
        "set": combo_high_recall,
        "is_baseline": False,
    },
    {
        "id": "combo_name4_addr_street",
        "name": "COMBO: Name Norm(4) + Addr Num & Street(3)",
        "set": combo_name4_addr_street,
        "is_baseline": False,
    },
    {
        "id": "combo_ultra_selective",
        "name": "COMBO: Ultra-Selective Compounds (Name3+AddrNum + Name3+Addr3 + Token4+AddrNum)",
        "set": combo_ultra_selective,
        "is_baseline": False,
    },
]


# ----------------------------------------------------------------------
# 6. EVALUATION AND DETAILED REPORTING
# ----------------------------------------------------------------------
print("\n" + "=" * 110)
print("BLOCKING EXPERIMENT 4: EVALUATION & COMPARATIVE REPORT")
print("=" * 110)
print(f"S1 sample size            : {len(sample_s1):,}")
print(f"Total S2 + S3 universe    : {total_universe:,}")
print(f"Total possible pairs      : {possible_pairs:,}")
print(f"Total Ground-Truth pairs  : {total_true:,}")
print(f"Baseline Candidate Count  : {baseline_cand_count:,}")
print(f"Baseline Recall           : {baseline_recall:.2%}")
print("-" * 110)

header = (
    f"{'Strategy / Blocking Key':<46} | "
    f"{'Candidates':<11} | "
    f"{'Recovered':<9} | "
    f"{'Recall':<7} | "
    f"{'Reduct.%':<9} | "
    f"{'Cand. vs Base':<13} | "
    f"{'Rec. vs Base'}"
)
print(header)
print("-" * len(header))

# Helper to format evaluation row
def print_eval_row(name, cand_set, is_baseline=False):
    found = true_pairs_set.intersection(cand_set)
    n_found = len(found)
    n_cand = len(cand_set)
    rec = (n_found / total_true) if total_true else 0.0
    red = (1.0 - (n_cand / possible_pairs)) * 100.0 if possible_pairs else 0.0

    if is_baseline:
        cand_vs_base = "BASELINE"
        rec_vs_base = "BASELINE"
    else:
        # Candidate delta vs baseline: positive means fewer candidates (good!)
        cand_diff_pct = ((n_cand - baseline_cand_count) / baseline_cand_count) * 100.0
        rec_diff_pct = (rec - baseline_recall) * 100.0
        cand_vs_base = f"{cand_diff_pct:+.1f}%"
        rec_vs_base = f"{rec_diff_pct:+.2f}%"

    print(
        f"{name:<46} | "
        f"{n_cand:<11,d} | "
        f"{n_found:<9,d} | "
        f"{rec:<7.2%} | "
        f"{red:<8.4f}% | "
        f"{cand_vs_base:<13} | "
        f"{rec_vs_base}"
    )


# 1. Print Individual Strategies Grouped by Investigation
current_cat = None
for strat in STRATEGY_METADATA:
    cat = strat["category"]
    if cat != current_cat:
        print(f"\n--- [{cat}] ---")
        current_cat = cat
    print_eval_row(strat["name"], candidate_sets[strat["id"]])

# 2. Print Multi-Key Combinations
print("\n" + "=" * 110)
print("MULTI-KEY ENSEMBLE COMBINATIONS VS BASELINE")
print("=" * 110)
for combo in COMBINED_STRATEGIES:
    print_eval_row(combo["name"], combo["set"], is_baseline=combo["is_baseline"])

print("=" * 110)
print("\nExecution complete. Experiment file blocking_exp4.py finished successfully.")
