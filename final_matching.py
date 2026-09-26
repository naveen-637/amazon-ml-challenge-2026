import os
import re
import time
import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"

THRESHOLD = 0.82
CHUNK_SIZE = 100_000

S1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
S2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
S3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def fast_norm(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return value.strip()


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("=" * 80)
print("FINAL BUSINESS ENTITY MATCHING PIPELINE")
print("=" * 80)

print("\nLoading Source 1...")

s1_cols = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    usecols=s1_cols
)

print(f"Source 1 records: {len(s1):,}")


# ============================================================
# NORMALIZE SOURCE 1
# ============================================================

print("Normalizing Source 1...")

s1["name_norm"] = s1["business_name"].map(fast_norm)
s1["address_norm"] = s1["business_address"].map(fast_norm)
s1["country_norm"] = s1["country"].map(fast_norm)

s1["name_key"] = (
    s1["name_norm"]
    .str.replace(" ", "", regex=False)
    .str[:4]
)

s1["address_key"] = (
    s1["address_norm"]
    .str.replace(" ", "", regex=False)
    .str[:6]
)


# ============================================================
# BUILD BLOCKING INDEX
# ============================================================

print("Building Source 1 blocking indexes...")

name_index = {}
address_index = {}

for idx, row in s1.iterrows():

    country = row["country_norm"]

    if row["name_key"]:
        key = country + "||" + row["name_key"]

        name_index.setdefault(key, []).append(idx)

    if row["address_key"]:
        key = country + "||" + row["address_key"]

        address_index.setdefault(key, []).append(idx)


# ============================================================
# RESULT STORAGE
# ============================================================

matches = {
    entity_id: set()
    for entity_id in s1["entity_id"]
}

candidate_pairs = set()

total_candidates = 0
total_scored = 0
total_matches = 0


# ============================================================
# PROCESS SOURCE 2 / SOURCE 3
# ============================================================

def process_source(file_path, source_name):

    global total_candidates
    global total_scored
    global total_matches

    print("\n" + "=" * 80)
    print(f"PROCESSING {source_name}")
    print("=" * 80)

    cols = [
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]

    chunk_number = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        usecols=cols,
        chunksize=CHUNK_SIZE
    ):

        chunk_number += 1

        print(
            f"\n{source_name} | Chunk {chunk_number} "
            f"| {len(chunk):,} rows"
        )

        # ----------------------------------------------------
        # Normalize chunk
        # ----------------------------------------------------

        chunk["name_norm"] = chunk["business_name"].map(fast_norm)
        chunk["address_norm"] = chunk["business_address"].map(fast_norm)
        chunk["country_norm"] = chunk["country"].map(fast_norm)

        chunk["name_key"] = (
            chunk["name_norm"]
            .str.replace(" ", "", regex=False)
            .str[:4]
        )

        chunk["address_key"] = (
            chunk["address_norm"]
            .str.replace(" ", "", regex=False)
            .str[:6]
        )

        # ----------------------------------------------------
        # Process each source row
        # ----------------------------------------------------

        for row in chunk.itertuples(index=False):

            country = row.country_norm

            if not country:
                continue

            candidate_s1_indices = set()

            # -----------------------------------------------
            # NAME BLOCK
            # -----------------------------------------------

            if row.name_key:

                key = country + "||" + row.name_key

                candidate_s1_indices.update(
                    name_index.get(key, [])
                )

            # -----------------------------------------------
            # ADDRESS BLOCK
            # -----------------------------------------------

            if row.address_key:

                key = country + "||" + row.address_key

                candidate_s1_indices.update(
                    address_index.get(key, [])
                )

            if not candidate_s1_indices:
                continue

            total_candidates += len(candidate_s1_indices)

            # -----------------------------------------------
            # SCORE CANDIDATES
            # -----------------------------------------------

            for s1_idx in candidate_s1_indices:

                s1row = s1.iloc[s1_idx]

                # Country must match
                if country != s1row["country_norm"]:
                    continue

                name_a = s1row["name_norm"]
                name_b = row.name_norm

                addr_a = s1row["address_norm"]
                addr_b = row.address_norm

                # -------------------------------------------
                # NAME SIMILARITY
                # -------------------------------------------

                if name_a and name_b:

                    name_sim = (
                        fuzz.token_sort_ratio(
                            name_a,
                            name_b
                        ) / 100.0
                    )

                else:
                    name_sim = 0.0

                # -------------------------------------------
                # ADDRESS SIMILARITY
                # -------------------------------------------

                if addr_a and addr_b:

                    addr_sim = (
                        fuzz.token_sort_ratio(
                            addr_a,
                            addr_b
                        ) / 100.0
                    )

                else:
                    addr_sim = 0.0

                total_scored += 1

                # -------------------------------------------
                # EXACT SIGNALS
                # -------------------------------------------

                name_exact = (
                    name_a != ""
                    and name_a == name_b
                )

                addr_exact = (
                    addr_a != ""
                    and addr_a == addr_b
                )

                # -------------------------------------------
                # BASE SCORE
                # -------------------------------------------

                score = (
                    0.60 * name_sim
                    +
                    0.40 * addr_sim
                )

                # Exact name is a very strong signal
                if name_exact:
                    score = max(
                        score,
                        0.95
                    )

                # Exact address provides a strong boost
                if addr_exact:
                    score = max(
                        score,
                        0.90
                    )

                # Both exact = essentially certain match
                if name_exact and addr_exact:
                    score = 1.0

                # -------------------------------------------
                # THRESHOLD
                # -------------------------------------------

                if score >= THRESHOLD:

                    s1_entity_id = s1row["entity_id"]
                    source_entity_id = row.entity_id

                    matches[s1_entity_id].add(
                        source_entity_id
                    )

                    candidate_pairs.add(
                        (
                            s1_entity_id,
                            source_entity_id
                        )
                    )

                    total_matches += 1

                else:
                    # Candidate was considered by the model,
                    # so it belongs in candidate_pairs.
                    candidate_pairs.add(
                        (
                            s1row["entity_id"],
                            row.entity_id
                        )
                    )

        print(
            f"Candidates considered so far: "
            f"{total_candidates:,}"
        )

        print(
            f"Scored so far: "
            f"{total_scored:,}"
        )

        print(
            f"Matches so far: "
            f"{total_matches:,}"
        )


# ============================================================
# RUN SOURCE 2
# ============================================================

start_time = time.time()

process_source(
    S2_FILE,
    "SOURCE 2"
)


# ============================================================
# RUN SOURCE 3
# ============================================================

process_source(
    S3_FILE,
    "SOURCE 3"
)


elapsed = time.time() - start_time


# ============================================================
# WRITE MATCHING RESULTS
# ============================================================

print("\n" + "=" * 80)
print("WRITING MATCHING RESULTS")
print("=" * 80)

matching_path = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)

with open(
    matching_path,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    f.write(
        "source1_entity_id\tmatched_entity_ids\n"
    )

    for entity_id in s1["entity_id"]:

        matched_ids = sorted(
            matches[entity_id]
        )

        f.write(
            f"{entity_id}\t"
            f"{','.join(matched_ids)}\n"
        )


# ============================================================
# WRITE CANDIDATE PAIRS
# ============================================================

print("Writing candidate pairs...")

candidate_path = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

with open(
    candidate_path,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    f.write(
        "source1_entity_id\tcandidate_entity_id\n"
    )

    for s1_id, candidate_id in sorted(
        candidate_pairs
    ):

        f.write(
            f"{s1_id}\t{candidate_id}\n"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL PIPELINE COMPLETED")
print("=" * 80)

print(
    f"Source 1 records        : {len(s1):,}"
)

print(
    f"Candidate pairs         : "
    f"{len(candidate_pairs):,}"
)

print(
    f"Scored candidates       : "
    f"{total_scored:,}"
)

print(
    f"Final matched pairs     : "
    f"{total_matches:,}"
)

print(
    f"Threshold               : "
    f"{THRESHOLD}"
)

print(
    f"Runtime                 : "
    f"{elapsed / 60:.2f} minutes"
)

print(
    f"\nCreated:"
)

print(
    f"  {matching_path}"
)

print(
    f"  {candidate_path}"
)

print("=" * 80)