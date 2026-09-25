import pandas as pd
import re
from rapidfuzz import fuzz

BASE = "dataset/train"

# Only load required columns
cols = ["entity_id", "business_name", "business_address", "country"]

s1 = pd.read_csv(f"{BASE}/train_source1.tsv", sep="\t", usecols=cols)
s2 = pd.read_csv(f"{BASE}/train_source2.tsv", sep="\t", usecols=cols)
s3 = pd.read_csv(f"{BASE}/train_source3.tsv", sep="\t", usecols=cols)
gt = pd.read_csv(f"{BASE}/train_ground_truth.tsv", sep="\t")


# -----------------------------
# Fast normalization
# -----------------------------

def fast_norm(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)

    return x.strip()


# -----------------------------
# Sample S1
# -----------------------------

sample_s1 = s1.head(1000).copy()
sample_ids = set(sample_s1["entity_id"])


# Normalize ONLY the 1000 S1 rows
sample_s1["name_norm"] = sample_s1["business_name"].map(fast_norm)
sample_s1["address_norm"] = sample_s1["business_address"].map(fast_norm)


# -----------------------------
# Create lightweight keys
# -----------------------------

sample_s1["name_key"] = (
    sample_s1["name_norm"]
    .str.replace(" ", "", regex=False)
    .str[:4]
)

sample_s1["address_key"] = (
    sample_s1["address_norm"]
    .str.replace(" ", "", regex=False)
    .str[:6]
)


# -----------------------------
# Ground truth
# -----------------------------

sample_gt = gt[
    gt["source1_entity_id"].isin(sample_ids)
]

true_pairs = set()

for _, row in sample_gt.iterrows():

    matched = row["matched_entity_ids"]

    if pd.isna(matched) or matched == "":
        continue

    for entity_id in matched.split(","):
        true_pairs.add(
            (row["source1_entity_id"], entity_id)
        )


# -----------------------------
# Process S2 + S3 in chunks
# -----------------------------

candidate_set = set()

source_files = [
    f"{BASE}/train_source2.tsv",
    f"{BASE}/train_source3.tsv"
]

for file in source_files:

    print("\nProcessing:", file)

    for chunk in pd.read_csv(
        file,
        sep="\t",
        usecols=cols,
        chunksize=250_000
    ):

        chunk["name_norm"] = chunk["business_name"].map(fast_norm)
        chunk["address_norm"] = chunk["business_address"].map(fast_norm)

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

        # Compare only rows sharing a country + name/address key
        for _, s1row in sample_s1.iterrows():

            country = str(s1row["country"])

            # NAME BLOCK
            if s1row["name_key"]:
                candidates = chunk[
                    (chunk["country"].astype(str) == country)
                    &
                    (chunk["name_key"] == s1row["name_key"])
                ]

                for _, r in candidates.iterrows():

                    candidate_set.add(
                        (s1row["entity_id"], r["entity_id"])
                    )

            # ADDRESS BLOCK
            if s1row["address_key"]:
                candidates = chunk[
                    (chunk["country"].astype(str) == country)
                    &
                    (chunk["address_key"] == s1row["address_key"])
                ]

                for _, r in candidates.iterrows():

                    candidate_set.add(
                        (s1row["entity_id"], r["entity_id"])
                    )


# -----------------------------
# Evaluation
# -----------------------------

found = true_pairs.intersection(candidate_set)

total_true = len(true_pairs)
total_found = len(found)

recall = (
    total_found / total_true
    if total_true
    else 0
)

possible_pairs = len(sample_s1) * (
    len(s2) + len(s3)
)

reduction = 1 - (
    len(candidate_set) / possible_pairs
)


print("\n==============================")
print("EXPERIMENT 3")
print("==============================")

print("S1 sample:", len(sample_s1))
print("Candidate pairs:", len(candidate_set))
print("Ground-truth pairs:", total_true)
print("True pairs recovered:", total_found)
print(f"Blocking recall: {recall:.2%}")
print(f"Candidate reduction: {reduction:.6%}")