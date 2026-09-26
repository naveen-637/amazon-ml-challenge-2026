import pandas as pd
import re
import time

BASE = "dataset/train"

cols = ["entity_id", "business_name", "business_address", "country"]

# -------------------------------------------------
# NORMALIZATION FUNCTIONS
# -------------------------------------------------

def fast_norm(x):
    """
    Original normalization function kept for reference/compatibility.
    Lowers text, replaces non-alphanumeric chars with space, and strips.
    """
    if pd.isna(x):
        return ""

    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)

    return x.strip()


def vectorized_clean_norm(series: pd.Series) -> pd.Series:
    """
    Vectorized string normalization using pandas C regex engine:
    1. Fill NA with empty string
    2. Lowercase all text
    3. Remove all non-alphanumeric characters (including spaces)
    Equivalent to fast_norm(x).replace(" ", "") but orders of magnitude faster.
    """
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


# -------------------------------------------------
# 1. LOAD S1 (1,000 SAMPLE)
# -------------------------------------------------
print("Loading S1 dataset...")
s1 = pd.read_csv(
    f"{BASE}/train_source1.tsv",
    sep="\t",
    usecols=cols
)

# Test strictly the first 1,000 S1 records
sample_s1 = s1.head(1000).copy()
sample_ids = set(sample_s1["entity_id"])


# -------------------------------------------------
# 2. VECTORIZED FEATURE & KEY GENERATION FOR S1
# -------------------------------------------------
print("Computing S1 blocking keys...")

s1_country = sample_s1["country"].fillna("").astype(str).str.strip()

# Normalized business name and address (alphanumeric only)
s1_name_norm = vectorized_clean_norm(sample_s1["business_name"])
s1_addr_norm = vectorized_clean_norm(sample_s1["business_address"])

# Raw address: trimmed and lowercased, preserving punctuation/spacing
s1_addr_raw = (
    sample_s1["business_address"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .str[:6]
)

# Retain original columns for backwards compatibility
sample_s1["name_norm"] = sample_s1["business_name"].map(fast_norm)
sample_s1["address_norm"] = sample_s1["business_address"].map(fast_norm)

# Generate blocking keys on S1
sample_s1["key_country_name_norm4"] = s1_country + "||" + s1_name_norm.str[:4]
sample_s1["key_country_addr_norm6"] = s1_country + "||" + s1_addr_norm.str[:6]
sample_s1["key_country_name_norm6"] = s1_country + "||" + s1_name_norm.str[:6]
sample_s1["key_country_addr_raw6"] = s1_country + "||" + s1_addr_raw


# -------------------------------------------------
# 3. BLOCKING STRATEGY CONFIGURATIONS
# -------------------------------------------------
# Multiple blocking keys requested:
# - country + first 4 normalized name characters
# - country + first 6 normalized address characters
# - country + first 6 normalized name characters
# - country + first 6 address characters (unnormalized)

STRATEGIES = [
    {
        "id": "strat_name_norm4",
        "name": "Country + Name Norm (4 chars)",
        "desc": "country + first 4 normalized name characters",
        "key_col": "key_country_name_norm4",
    },
    {
        "id": "strat_addr_norm6",
        "name": "Country + Address Norm (6 chars)",
        "desc": "country + first 6 normalized address characters",
        "key_col": "key_country_addr_norm6",
    },
    {
        "id": "strat_name_norm6",
        "name": "Country + Name Norm (6 chars)",
        "desc": "country + first 6 normalized name characters",
        "key_col": "key_country_name_norm6",
    },
    {
        "id": "strat_addr_raw6",
        "name": "Country + Address Raw (6 chars)",
        "desc": "country + first 6 address characters (unnormalized/trimmed)",
        "key_col": "key_country_addr_raw6",
    },
]

# Pre-build lookup dictionaries for S1: key -> list of s1_ids
# Because S1 only has 1,000 records, each strategy index contains <= 1,000 keys.
s1_lookup = {}
s1_valid_keys = {}

for strat in STRATEGIES:
    col = strat["key_col"]
    lookup = {}
    for s1_id, k in zip(sample_s1["entity_id"], sample_s1[col]):
        # Ignore empty key values (i.e. where key prefix was empty)
        if k and not k.endswith("||"):
            lookup.setdefault(k, []).append(s1_id)
    s1_lookup[strat["id"]] = lookup
    s1_valid_keys[strat["id"]] = set(lookup.keys())


# -------------------------------------------------
# 4. LOAD GROUND TRUTH (FOR S1 SAMPLE)
# -------------------------------------------------
print("Loading ground truth for evaluation...")
true_pairs = set()

for chunk in pd.read_csv(
    f"{BASE}/train_ground_truth.tsv",
    sep="\t",
    chunksize=250_000
):
    matched_gt = chunk[
        chunk["source1_entity_id"].isin(sample_ids)
    ]

    for row in matched_gt.itertuples(index=False):
        matched = getattr(row, "matched_entity_ids", "")
        if pd.isna(matched) or not matched:
            continue

        s1_id = getattr(row, "source1_entity_id")
        for entity_id in str(matched).split(","):
            entity_id = entity_id.strip()
            if entity_id:
                true_pairs.add((s1_id, entity_id))

print(f"S1 sample records: {len(sample_s1)}")
print(f"Ground-truth pairs to recover: {len(true_pairs)}")


# -------------------------------------------------
# 5. FAST CHUNKED BLOCKING OVER S2 AND S3
# -------------------------------------------------
# Candidate sets for each individual strategy
candidate_sets = {strat["id"]: set() for strat in STRATEGIES}

source_files = [
    f"{BASE}/train_source2.tsv",
    f"{BASE}/train_source3.tsv",
]

s2_count = 0
s3_count = 0
start_time = time.time()

print("\n=================================================")
print("RUNNING FAST BLOCKING EXPERIMENTS (S2 & S3 CHUNKS)")
print("=================================================")

for file in source_files:
    is_s2 = "source2" in file
    source_name = "Source 2" if is_s2 else "Source 3"
    print(f"\nProcessing {source_name} ({file}) in chunks...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file,
            sep="\t",
            usecols=cols,
            chunksize=250_000
        ),
        start=1
    ):
        chunk_len = len(chunk)
        if is_s2:
            s2_count += chunk_len
        else:
            s3_count += chunk_len

        print(f"  -> Chunk {chunk_number:2d}: {chunk_len:,} rows processed", end="\r")

        # --- VECTORIZED KEY COMPUTATION (No iterrows) ---
        chunk_country = chunk["country"].fillna("").astype(str).str.strip()

        # Vectorized normalized name and address
        chunk_name_norm = vectorized_clean_norm(chunk["business_name"])
        chunk_addr_norm = vectorized_clean_norm(chunk["business_address"])

        # Vectorized raw address (first 6 chars, lowercase trimmed)
        chunk_addr_raw = (
            chunk["business_address"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str[:6]
        )

        chunk["key_country_name_norm4"] = chunk_country + "||" + chunk_name_norm.str[:4]
        chunk["key_country_addr_norm6"] = chunk_country + "||" + chunk_addr_norm.str[:6]
        chunk["key_country_name_norm6"] = chunk_country + "||" + chunk_name_norm.str[:6]
        chunk["key_country_addr_raw6"] = chunk_country + "||" + chunk_addr_raw

        entity_ids = chunk["entity_id"].values

        # --- VECTORIZED CANDIDATE MATCHING (No iterrows) ---
        # Instead of scanning the chunk with iterrows or building dictionaries
        # for all 250k rows, we filter in C using .isin(s1_valid_keys).
        # Only the tiny fraction (<0.1%) of rows matching S1 keys are processed.
        for strat in STRATEGIES:
            strat_id = strat["id"]
            col = strat["key_col"]
            valid_keys = s1_valid_keys[strat_id]
            lookup = s1_lookup[strat_id]

            key_series = chunk[col]
            mask = key_series.isin(valid_keys)

            if mask.any():
                matched_s2_ids = entity_ids[mask]
                matched_keys = key_series[mask].values
                target_cand_set = candidate_sets[strat_id]

                for s2_id, k in zip(matched_s2_ids, matched_keys):
                    for s1_id in lookup[k]:
                        target_cand_set.add((s1_id, s2_id))

    print()  # Advance past chunk progress carriage return

elapsed = time.time() - start_time
print(f"\nChunk processing completed in {elapsed:.2f} seconds.")


# -------------------------------------------------
# 6. EVALUATION AND REPORTING
# -------------------------------------------------
total_s2_s3 = s2_count + s3_count
possible_pairs = len(sample_s1) * total_s2_s3
total_true = len(true_pairs)

print("\n" + "=" * 80)
print("BLOCKING EXPERIMENT EVALUATION RESULTS")
print("=" * 80)
print(f"S1 sample records   : {len(sample_s1):,}")
print(f"Total S2 records    : {s2_count:,}")
print(f"Total S3 records    : {s3_count:,}")
print(f"Total universe rows : {total_s2_s3:,}")
print(f"Total possible pairs: {possible_pairs:,}")
print(f"Ground-truth pairs  : {total_true:,}")
print("-" * 80)

# Build combined strategy sets
baseline_combined = candidate_sets["strat_name_norm4"].union(
    candidate_sets["strat_addr_norm6"]
)
all_combined = set().union(*candidate_sets.values())

# Summary report table
header = f"{'Strategy / Blocking Key':<46} | {'Candidates':<11} | {'Recovered':<10} | {'Recall':<8} | {'Reduction':<12}"
print(header)
print("-" * len(header))

report_items = []

for strat in STRATEGIES:
    cand_set = candidate_sets[strat["id"]]
    found = true_pairs.intersection(cand_set)
    n_found = len(found)
    n_cand = len(cand_set)
    rec = (n_found / total_true) if total_true else 0
    red = (1 - (n_cand / possible_pairs)) if possible_pairs else 0

    report_items.append((strat["name"], n_cand, n_found, rec, red))
    print(f"{strat['name']:<46} | {n_cand:<11,d} | {n_found:<10,d} | {rec:<8.2%} | {red:<12.6%}")

# Also report combined strategies
for label, c_set in [
    ("COMBINED: Name Norm(4) + Addr Norm(6) [Base]", baseline_combined),
    ("COMBINED: Union of All 4 Strategies", all_combined),
]:
    found = true_pairs.intersection(c_set)
    n_found = len(found)
    n_cand = len(c_set)
    rec = (n_found / total_true) if total_true else 0
    red = (1 - (n_cand / possible_pairs)) if possible_pairs else 0
    print(f"{label:<46} | {n_cand:<11,d} | {n_found:<10,d} | {rec:<8.2%} | {red:<12.6%}")

print("=" * 80)