import pandas as pd
import re
import unicodedata
from rapidfuzz import process, fuzz

BASE = "dataset/train"

s1 = pd.read_csv(f"{BASE}/train_source1.tsv", sep="\t")
s2 = pd.read_csv(f"{BASE}/train_source2.tsv", sep="\t")
s3 = pd.read_csv(f"{BASE}/train_source3.tsv", sep="\t")
gt = pd.read_csv(f"{BASE}/train_ground_truth.tsv", sep="\t")


def normalize_name(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def make_block_key(country, name):
    """
    Country + first 4 characters of normalized name.
    """
    name = normalize_name(name)

    if not name:
        return None

    return (str(country), name[:4])


# --------------------------------
# Sample
# --------------------------------

sample_s1 = s1.head(1000).copy()
sample_ids = set(sample_s1["entity_id"])


# --------------------------------
# Normalize
# --------------------------------

sample_s1["name_norm"] = sample_s1["business_name"].apply(
    normalize_name
)

s2["name_norm"] = s2["business_name"].apply(normalize_name)
s3["name_norm"] = s3["business_name"].apply(normalize_name)


# --------------------------------
# Build block index
# --------------------------------

block_index = {}


for entity_id, country, name in zip(
    s2["entity_id"],
    s2["country"],
    s2["name_norm"]
):
    if name:
        key = (str(country), name[:4])

        block_index.setdefault(key, []).append(
            (entity_id, name)
        )


for entity_id, country, name in zip(
    s3["entity_id"],
    s3["country"],
    s3["name_norm"]
):
    if name:
        key = (str(country), name[:4])

        block_index.setdefault(key, []).append(
            (entity_id, name)
        )


# --------------------------------
# Candidate generation
# --------------------------------

candidate_set = set()

for _, row in sample_s1.iterrows():

    name = row["name_norm"]

    if not name:
        continue

    key = (
        str(row["country"]),
        name[:4]
    )

    candidates = block_index.get(key, [])

    # Extract names
    choices = {
        entity_id: candidate_name
        for entity_id, candidate_name in candidates
    }

    # Find similar names
    matches = process.extract(
        name,
        choices,
        scorer=fuzz.ratio,
        score_cutoff=70,
        limit=50
    )

    for _, score, entity_id in matches:

        candidate_set.add(
            (row["entity_id"], entity_id)
        )


# --------------------------------
# Ground truth
# --------------------------------

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


# --------------------------------
# Evaluation
# --------------------------------

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


print("S1 sample:", len(sample_s1))
print("Candidate pairs:", len(candidate_set))

print("\nGround-truth pairs:", total_true)
print("True pairs recovered:", total_found)

print(f"Blocking recall: {recall:.2%}")
print(f"Candidate reduction: {reduction:.6%}")