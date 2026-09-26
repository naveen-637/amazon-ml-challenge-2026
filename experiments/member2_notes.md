# Member 2: Exploratory Feature Analysis Notes

## 1. Overview & What Was Tested
This investigation evaluates the discriminative ability and statistical distributions of string similarity and country matching features strictly on **True Match Pairs** extracted from the Ground Truth dataset, sampled across **1,000 Source 1 (S1) records**.

The goal is exploratory: to understand how features behave under real-world noise (typos, abbreviations, legal suffixes, missing address tokens) before training or tuning the final matching classifier.

### Features Computed:
- **`name_ratio` (`fuzz.ratio`)**: Standard Levenshtein distance ratio between business names.
- **`name_token_set_ratio` (`fuzz.token_set_ratio`)**: Token set similarity on business names (handles token reordering, subset matching, and legal suffixes).
- **`address_ratio` (`fuzz.ratio`)**: Levenshtein distance ratio between addresses.
- **`address_token_set_ratio` (`fuzz.token_set_ratio`)**: Token set similarity on addresses (handles partial street addresses, missing state/zip tokens).
- **`country_match`**: Exact boolean indicator for matching country codes (`1` if match, `0` otherwise).
- **`combined_score`**: Simple weighted exploratory score (`0.6 * name_token_set_ratio + 0.4 * address_token_set_ratio`).

---

## 2. Feature Distribution Across True Matches (Summary)

| Feature | Mean | Std Dev | Median | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Name Ratio (`fuzz.ratio`)** | 88.17 | 15.62 | 91.30 | 27.27 | 100.00 |
| **Name Token-Set Ratio** | 93.55 | 14.51 | 100.00 | 27.27 | 100.00 |
| **Address Ratio (`fuzz.ratio`)** | 85.49 | 18.72 | 94.87 | 44.74 | 100.00 |
| **Address Token-Set Ratio** | 90.23 | 13.08 | 94.87 | 47.46 | 100.00 |
| **Country Exact Match** | 1.0000 | 0.0000 | 1.00 | 1 | 1 |
| **Exploratory Combined Score** | 92.22 | 9.10 | 94.29 | 56.36 | 100.00 |

---

## 3. Key Observations & Feature Usefulness

1. **`fuzz.token_set_ratio` significantly outperforms `fuzz.ratio` on Noisy Text:**
   - Standard Levenshtein ratio drops sharply when legal suffixes differ (e.g. *"Private Limited"* vs *"Pvt Ltd"*) or when street components are abbreviated (*"Street"* vs *"St"*).
   - `token_set_ratio` isolates intersecting tokens and ignores reordering or extraneous legal tokens, achieving substantially higher scores on true positive pairs.

2. **Address Variations Require Token-Based / Partial Matching:**
   - Real-world address entries often omit the postal code, state, or landmark, or truncate after the street line.
   - While `address_ratio` has high variance, `address_token_set_ratio` reliably anchors matching entities that share street names and building numbers.

3. **Country Exact Match is a Hard Constraint:**
   - True matches consistently belong to the same country. Country filtering serves as an essential hard filter or high-weight feature to prevent cross-country false positives.

4. **Name vs. Address Weighting:**
   - Business names provide the primary anchor for entity identity, making the `0.6` name + `0.4` address weighting a strong baseline for initial candidate ranking.

---

## 4. Limitations of this Exploratory Analysis

- **True Matches Only (No Negative Pairs):** This script evaluates features exclusively on positive ground truth pairs. It does not measure the false positive rate on hard negative distractors (which will be evaluated in the full candidate blocking & classification stage).
- **No Threshold Optimization for $F_{0.5}$:** The exploratory `combined_score` uses static weights (0.6 / 0.4) without tuning the decision boundary to optimize the competition precision-weighted metric ($F_{0.5}$).
- **Sample Size:** Evaluated on a sample of 1,000 S1 records to maintain fast iteration and memory efficiency.
- **Not a Final Matching Model:** This script serves solely as feature exploration to inform model architecture, feature engineering, and classifier design.
