"""
Final Entity Matching Pipeline -- Amazon ML Challenge 2026
==========================================================
Reads test_source1.tsv, test_source2.tsv, test_source3.tsv in batches/chunks
and produces:
  output/matching_results.tsv   -- final matched pairs (scored by leaderboard)
  output/candidate_pairs.tsv    -- all blocked candidate pairs

Pipeline Architecture
---------------------
1. Collect all test S1 entity IDs (single pass, entity_id column only).
2. Process S1 in batches of S1_BATCH_SIZE (default: 10,000).
   For each S1 batch:
     a. Build blocking index: country + name[:4] and country + addr[:6].
     b. Stream test_source2.tsv then test_source3.tsv in chunks of CHUNK_SIZE.
     c. For each chunk row matching the blocking index, score the pair.
     d. If score >= THRESHOLD, record as a match.
     e. Append to output files immediately (low memory footprint).
3. After all S1 batches complete, verify output row count == len(test_S1).

Memory Strategy
---------------
- Never load all S1 into memory at once.
- Never materialise millions of candidate pairs in Python sets.
- Each S1 batch index holds ≤ S1_BATCH_SIZE * avg_block_size keys.
- Accepted pairs are written to disk immediately and discarded from RAM.
- Peak RAM target: ≤ 70% of system memory.

Tuning Parameters (top of script)
----------------------------------
THRESHOLD      : Combined score cut-off (validated: 0.82 → Macro F0.5 ≈ 0.685)
S1_BATCH_SIZE  : S1 records per batch (10_000 is safe for 16 GB RAM)
CHUNK_SIZE     : S2/S3 rows per read chunk (100_000)
SAVE_CANDIDATES: True = also write candidate_pairs.tsv (slightly slower)
"""

import os
import sys
import time
import gc
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from rapidfuzz import fuzz
except ImportError:
    print("ERROR: rapidfuzz not installed. Run: pip install rapidfuzz")
    sys.exit(1)

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ======================================================================
# TUNING PARAMETERS -- adjust here
# ======================================================================
THRESHOLD      = 0.82          # matching score cut-off (F0.5-optimised on validation)
S1_BATCH_SIZE  = 10_000        # S1 records per processing batch
CHUNK_SIZE     = 100_000       # rows per S2/S3 read chunk
SAVE_CANDIDATES = True         # also produce candidate_pairs.tsv

# ======================================================================
# PATH DISCOVERY
# ======================================================================
def _find_dir(candidates):
    for p in candidates:
        if Path(p).exists():
            return str(p)
    return None

TEST_DIR = _find_dir([
    "dataset/test",
    "student_resource/dataset/test",
    str(Path(__file__).resolve().parent.parent / "dataset" / "test"),
    str(Path(__file__).resolve().parent / "dataset" / "test"),
])

OUT_DIR = _find_dir([
    "output",
    "student_resource/output",
    str(Path(__file__).resolve().parent.parent / "output"),
])

if TEST_DIR is None:
    print("ERROR: Cannot locate dataset/test/ directory. Run from student_resource/.")
    sys.exit(1)

if OUT_DIR is None:
    # Create output directory relative to script location
    OUT_DIR = str(Path(__file__).resolve().parent.parent / "output")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

TEST_S1  = str(Path(TEST_DIR) / "test_source1.tsv")
TEST_S2  = str(Path(TEST_DIR) / "test_source2.tsv")
TEST_S3  = str(Path(TEST_DIR) / "test_source3.tsv")
OUT_MATCH = str(Path(OUT_DIR) / "matching_results.tsv")
OUT_CAND  = str(Path(OUT_DIR) / "candidate_pairs.tsv")

cols = ["entity_id", "business_name", "business_address", "country"]

# ======================================================================
# NORMALISATION HELPERS
# ======================================================================

def norm_clean(series: pd.Series) -> pd.Series:
    """Alphanumeric lowercase string (no spaces) -- used for blocking keys & exact match."""
    return (
        series.fillna("").astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def norm_words(series: pd.Series) -> pd.Series:
    """Lowercase with normalised whitespace -- used for RapidFuzz token scoring."""
    return (
        series.fillna("").astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ======================================================================
# RAM MONITOR (optional)
# ======================================================================

def ram_pct():
    if _PSUTIL:
        return psutil.virtual_memory().percent
    return -1.0


def check_ram(warn_pct=70.0, stop_pct=80.0):
    pct = ram_pct()
    if pct >= stop_pct:
        print(f"\n[WARN]  RAM at {pct:.1f}% -- STOPPING to protect system. Reduce S1_BATCH_SIZE.")
        sys.exit(1)
    if pct >= warn_pct:
        print(f"\n[WARN]  RAM at {pct:.1f}% (warning threshold {warn_pct}%)")
    return pct


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    print("=" * 80)
    print("FINAL ENTITY MATCHING PIPELINE -- Amazon ML Challenge 2026")
    print("=" * 80)
    print(f"Test dir        : {TEST_DIR}")
    print(f"Output dir      : {OUT_DIR}")
    print(f"Threshold       : {THRESHOLD}")
    print(f"S1 batch size   : {S1_BATCH_SIZE:,}")
    print(f"Chunk size      : {CHUNK_SIZE:,}")
    print(f"Save candidates : {SAVE_CANDIDATES}")
    print()

    # ------------------------------------------------------------------
    # STEP 1: Read all test S1 entity IDs (lightweight -- entity_id only)
    # ------------------------------------------------------------------
    print("Step 1: Reading test S1 entity IDs...")
    all_s1_ids = []
    for chunk in pd.read_csv(TEST_S1, sep="\t", usecols=["entity_id"], chunksize=250_000):
        all_s1_ids.extend(chunk["entity_id"].tolist())
    total_s1 = len(all_s1_ids)
    print(f"  Total S1 entities: {total_s1:,}")

    # ------------------------------------------------------------------
    # STEP 2: Prepare output files (write headers)
    # ------------------------------------------------------------------
    print("Step 2: Initialising output files...")
    with open(OUT_MATCH, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
    if SAVE_CANDIDATES:
        with open(OUT_CAND, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")

    # ------------------------------------------------------------------
    # STEP 3: Process S1 in batches
    # ------------------------------------------------------------------
    n_batches = (total_s1 + S1_BATCH_SIZE - 1) // S1_BATCH_SIZE
    print(f"Step 3: Processing {n_batches} S1 batches ({S1_BATCH_SIZE:,} records each)...")
    pipeline_start = time.time()

    # Running totals for progress reporting
    total_pairs_written = 0
    total_cands_written  = 0

    for batch_idx in range(n_batches):
        batch_start = time.time()
        s1_start = batch_idx * S1_BATCH_SIZE
        s1_end   = min(s1_start + S1_BATCH_SIZE, total_s1)
        batch_ids = all_s1_ids[s1_start:s1_end]

        print(f"\n  --- Batch {batch_idx + 1}/{n_batches}  "
              f"[S1 rows {s1_start:,}–{s1_end:,}]  "
              f"RAM: {ram_pct():.1f}% ---")

        # Load this S1 batch from test_source1
        # We use skiprows to efficiently skip already-processed rows.
        # skiprows=range(1, s1_start+1) skips rows (0-indexed after header).
        skip = range(1, s1_start + 1) if s1_start > 0 else None
        s1_batch = pd.read_csv(
            TEST_S1, sep="\t", usecols=cols,
            skiprows=skip, nrows=S1_BATCH_SIZE, header=0
        )

        # Vectorised feature extraction for S1 batch
        s1_country   = s1_batch["country"].fillna("").astype(str).str.strip().str.upper().values
        s1_name_c    = norm_clean(s1_batch["business_name"]).values
        s1_addr_c    = norm_clean(s1_batch["business_address"]).values
        s1_name_w    = norm_words(s1_batch["business_name"]).values
        s1_addr_w    = norm_words(s1_batch["business_address"]).values
        s1_ids_batch = s1_batch["entity_id"].values

        # Build blocking index for this batch
        lookup_name4: dict = {}
        lookup_addr6: dict = {}
        for i in range(len(s1_batch)):
            c  = s1_country[i]
            nc = s1_name_c[i]
            ac = s1_addr_c[i]
            kn = c + "||" + nc[:4]
            ka = c + "||" + ac[:6]
            if kn and not kn.endswith("||"):
                lookup_name4.setdefault(kn, []).append(i)
            if ka and not ka.endswith("||"):
                lookup_addr6.setdefault(ka, []).append(i)

        valid_name4 = set(lookup_name4)
        valid_addr6 = set(lookup_addr6)

        # Per-entity accumulators for this batch (small: only batch_size entries)
        # matches[i] = set of matched target entity IDs
        # cands[i]   = set of candidate target entity IDs
        matches: dict = {i: [] for i in range(len(s1_batch))}
        cands:   dict = {i: [] for i in range(len(s1_batch))} if SAVE_CANDIDATES else {}

        # Score S2 and S3
        for src_file, src_name in [(TEST_S2, "S2"), (TEST_S3, "S3")]:
            for chunk_num, chunk in enumerate(
                pd.read_csv(src_file, sep="\t", usecols=cols, chunksize=CHUNK_SIZE), 1
            ):
                c_country = chunk["country"].fillna("").astype(str).str.strip().str.upper()
                c_name_c  = norm_clean(chunk["business_name"])
                c_addr_c  = norm_clean(chunk["business_address"])
                c_name_w  = norm_words(chunk["business_name"])
                c_addr_w  = norm_words(chunk["business_address"])
                c_ids     = chunk["entity_id"].values

                k_name = c_country + "||" + c_name_c.str[:4]
                k_addr = c_country + "||" + c_addr_c.str[:6]

                mask = k_name.isin(valid_name4) | k_addr.isin(valid_addr6)
                hit_idx = np.where(mask.values)[0]

                if len(hit_idx) == 0:
                    continue

                # Convert series to arrays for fast indexing
                c_country_a = c_country.values
                c_name_c_a  = c_name_c.values
                c_addr_c_a  = c_addr_c.values
                c_name_w_a  = c_name_w.values
                c_addr_w_a  = c_addr_w.values
                k_name_a    = k_name.values
                k_addr_a    = k_addr.values

                for ri in hit_idx:
                    tid      = c_ids[ri]
                    t_ctry   = c_country_a[ri]
                    t_name_c = c_name_c_a[ri]
                    t_addr_c = c_addr_c_a[ri]
                    t_name_w = c_name_w_a[ri]
                    t_addr_w = c_addr_w_a[ri]
                    t_kn     = k_name_a[ri]
                    t_ka     = k_addr_a[ri]

                    # Collect unique S1 batch indices for this candidate
                    s1_set: set = set()
                    if t_kn in valid_name4:
                        for si in lookup_name4[t_kn]:
                            s1_set.add(si)
                    if t_ka in valid_addr6:
                        for si in lookup_addr6[t_ka]:
                            s1_set.add(si)

                    for si in s1_set:
                        # Hard filter: country must match
                        if s1_country[si] != t_ctry:
                            continue

                        # Record as candidate
                        if SAVE_CANDIDATES:
                            cands[si].append(tid)

                        # Feature 1: name exact match
                        ne = (s1_name_c[si] == t_name_c)
                        # Feature 3: address exact match
                        ae = (s1_addr_c[si] == t_addr_c)

                        # Feature 2: name RapidFuzz similarity
                        ns = 1.0 if ne else fuzz.token_sort_ratio(s1_name_w[si], t_name_w) / 100.0
                        # Feature 4: address RapidFuzz similarity
                        as_ = 1.0 if ae else fuzz.token_sort_ratio(s1_addr_w[si], t_addr_w) / 100.0

                        # Feature 6: combined score (60% name, 40% address)
                        score = 0.60 * ns + 0.40 * as_

                        # Boost when one signal is an exact match
                        if ne and as_ >= 0.50:
                            score = max(score, 0.50 + 0.50 * as_)
                        elif ae and ns >= 0.50:
                            score = max(score, 0.40 + 0.60 * ns)

                        if score >= THRESHOLD:
                            matches[si].append(tid)

                print(
                    f"    {src_name} chunk {chunk_num:3d} "
                    f"| hits: {len(hit_idx):,} "
                    f"| matches so far: {sum(len(v) for v in matches.values()):,}",
                    end="\r"
                )
            print()  # newline after carriage-return progress

        # ------------------------------------------------------------------
        # Write this batch to output files
        # ------------------------------------------------------------------
        with open(OUT_MATCH, "a", encoding="utf-8") as fm:
            for i, s1_id in enumerate(s1_ids_batch):
                m_ids = matches[i]
                # Deduplicate while preserving order
                seen = set(); deduped = []
                for x in m_ids:
                    if x not in seen:
                        seen.add(x); deduped.append(x)
                fm.write(f"{s1_id}\t{','.join(deduped)}\n")
                total_pairs_written += len(deduped)

        if SAVE_CANDIDATES:
            with open(OUT_CAND, "a", encoding="utf-8") as fc:
                for i, s1_id in enumerate(s1_ids_batch):
                    c_ids_list = cands[i]
                    seen = set(); deduped = []
                    for x in c_ids_list:
                        if x not in seen:
                            seen.add(x); deduped.append(x)
                    fc.write(f"{s1_id}\t{','.join(deduped)}\n")
                    total_cands_written += len(deduped)

        batch_elapsed = time.time() - batch_start
        total_elapsed = time.time() - pipeline_start
        remaining_batches = n_batches - batch_idx - 1
        eta_secs = (total_elapsed / (batch_idx + 1)) * remaining_batches

        print(
            f"  Batch {batch_idx+1} done in {batch_elapsed:.1f}s "
            f"| Total matches written: {total_pairs_written:,} "
            f"| ETA: {eta_secs/60:.1f} min | RAM: {ram_pct():.1f}%"
        )

        # Free batch memory explicitly
        del s1_batch, s1_country, s1_name_c, s1_addr_c, s1_name_w, s1_addr_w
        del lookup_name4, lookup_addr6, valid_name4, valid_addr6
        del matches
        if SAVE_CANDIDATES:
            del cands
        gc.collect()

        check_ram(warn_pct=70.0, stop_pct=82.0)

    # ------------------------------------------------------------------
    # STEP 4: Verify output
    # ------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*80}")
    print(f"Pipeline completed in {total_elapsed/60:.1f} minutes.")
    print(f"Total match predictions written : {total_pairs_written:,}")
    if SAVE_CANDIDATES:
        print(f"Total candidate pairs written   : {total_cands_written:,}")

    print("\nStep 4: Verifying output file row count...")
    match_rows = 0
    with open(OUT_MATCH, "r", encoding="utf-8") as f:
        for line in f:
            match_rows += 1
    match_rows -= 1  # subtract header
    print(f"  matching_results.tsv rows (excl. header): {match_rows:,}")
    print(f"  Expected (= test S1 entities)           : {total_s1:,}")
    if match_rows == total_s1:
        print("  [OK] Row count matches. Output file looks correct.")
    else:
        print(f"  ❌ MISMATCH: expected {total_s1}, got {match_rows}. Check for errors.")

    print(f"\nOutput files saved to: {OUT_DIR}")
    print(f"  {OUT_MATCH}")
    if SAVE_CANDIDATES:
        print(f"  {OUT_CAND}")
    print("\nRun validation before submitting:")
    print("  python utils/validate_submission.py \\")
    print("    --matching output/matching_results.tsv \\")
    print("    --candidate output/candidate_pairs.tsv \\")
    print("    --test-dir dataset/test")


# ======================================================================
# SMALL SAMPLE TEST MODE
# ======================================================================

def sample_test(n_s1=100, n_chunk=50_000):
    """
    Quick smoke test on tiny sample: n_s1 S1 records × 50k S2/S3 rows.
    Prints metrics without writing output files.
    """
    import psutil
    proc_psutil = psutil.Process(os.getpid()) if _PSUTIL else None

    print(f"\n{'='*70}")
    print(f"SAMPLE TEST: {n_s1} S1 records × {n_chunk:,} S2/S3 rows")
    print(f"{'='*70}")

    s1_batch = pd.read_csv(TEST_S1, sep="\t", usecols=cols, nrows=n_s1)
    s1_country = s1_batch["country"].fillna("").astype(str).str.strip().str.upper().values
    s1_name_c  = norm_clean(s1_batch["business_name"]).values
    s1_addr_c  = norm_clean(s1_batch["business_address"]).values
    s1_name_w  = norm_words(s1_batch["business_name"]).values
    s1_addr_w  = norm_words(s1_batch["business_address"]).values
    s1_ids_b   = s1_batch["entity_id"].values

    lookup_name4: dict = {}
    lookup_addr6: dict = {}
    for i in range(n_s1):
        c = s1_country[i]; nc = s1_name_c[i]; ac = s1_addr_c[i]
        kn = c + "||" + nc[:4]; ka = c + "||" + ac[:6]
        if kn and not kn.endswith("||"): lookup_name4.setdefault(kn, []).append(i)
        if ka and not ka.endswith("||"): lookup_addr6.setdefault(ka, []).append(i)
    valid_name4 = set(lookup_name4); valid_addr6 = set(lookup_addr6)

    total_hits = 0; total_scored = 0; total_accepted = 0
    t0 = time.time()

    for src_file, src_name in [(TEST_S2, "S2"), (TEST_S3, "S3")]:
        chunk = pd.read_csv(src_file, sep="\t", usecols=cols, nrows=n_chunk)
        c_ctry = chunk["country"].fillna("").astype(str).str.strip().str.upper()
        c_nc   = norm_clean(chunk["business_name"])
        c_ac   = norm_clean(chunk["business_address"])
        c_nw   = norm_words(chunk["business_name"])
        c_aw   = norm_words(chunk["business_address"])
        c_ids  = chunk["entity_id"].values

        k_name = c_ctry + "||" + c_nc.str[:4]
        k_addr = c_ctry + "||" + c_ac.str[:6]
        mask   = k_name.isin(valid_name4) | k_addr.isin(valid_addr6)
        hidx   = np.where(mask.values)[0]
        total_hits += len(hidx)

        c_ctry_a = c_ctry.values; c_nc_a = c_nc.values; c_ac_a = c_ac.values
        c_nw_a = c_nw.values; c_aw_a = c_aw.values
        k_name_a = k_name.values; k_addr_a = k_addr.values

        for ri in hidx:
            s1_set = set()
            kn = k_name_a[ri]; ka = k_addr_a[ri]
            if kn in valid_name4:
                for si in lookup_name4[kn]: s1_set.add(si)
            if ka in valid_addr6:
                for si in lookup_addr6[ka]: s1_set.add(si)
            for si in s1_set:
                if s1_country[si] != c_ctry_a[ri]: continue
                total_scored += 1
                ne = (s1_name_c[si] == c_nc_a[ri])
                ae = (s1_addr_c[si] == c_ac_a[ri])
                ns = 1.0 if ne else fuzz.token_sort_ratio(s1_name_w[si], c_nw_a[ri]) / 100.0
                as_ = 1.0 if ae else fuzz.token_sort_ratio(s1_addr_w[si], c_aw_a[ri]) / 100.0
                score = 0.60 * ns + 0.40 * as_
                if ne and as_ >= 0.50: score = max(score, 0.50 + 0.50 * as_)
                elif ae and ns >= 0.50: score = max(score, 0.40 + 0.60 * ns)
                if score >= THRESHOLD: total_accepted += 1

    elapsed = time.time() - t0
    ram = proc_psutil.memory_info().rss / 1e6 if proc_psutil else -1

    print(f"  Time         : {elapsed:.2f}s")
    print(f"  Blocking hits: {total_hits:,}")
    print(f"  Pairs scored : {total_scored:,}")
    print(f"  Accepted     : {total_accepted:,} (threshold {THRESHOLD})")
    print(f"  RAM used     : {ram:.0f} MB")
    print(f"  SAFE TO RUN FULL PIPELINE: {'YES [OK]' if ram < 8000 else 'CHECK RAM [WARN]'}")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    if "--sample-test" in sys.argv:
        sample_test()
    else:
        main()
