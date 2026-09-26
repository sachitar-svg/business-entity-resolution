# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** TriForge
**Team Members:** A R Yashaswi, Jessa Mariya Joe, Sachita R
**Submission Date:** 27

---

## 1. Executive Summary

We developed a scalable business entity resolution pipeline for identifying corresponding business records across Source 1, Source 2, and Source 3. The solution combines Unicode-safe preprocessing, token-frequency-based candidate blocking, disk-backed SQLite/FTS5 candidate generation, pairwise feature engineering, supervised matching, hard-negative mining, and a tree-based matching model.

The current development pipeline uses the frozen V2 blocker and Tree V3 matching model. On the 200-S1 validation holdout, Tree V3 achieved precision 0.9231, recall 0.7192, Pair F0.5 0.8736, and Macro F0.5 0.8145 at threshold 0.950.

---

## 2. Methodology

### 2.1 Problem Analysis

The task contains three independent business-record sources without a universal shared identifier. Source 1 is the deduplicated reference source, while Source 2 and Source 3 contain records that may correspond to Source 1 entities.

A Source 1 entity may correspond to zero, one, or multiple records from Source 2 and Source 3.

The development data contains:

- Source 1: 2,206,821 rows
- Source 2: 5,034,616 rows
- Source 3: 5,285,603 rows
- Ground Truth: 2,206,821 Source 1 rows

Observed data issues included inconsistent business names, abbreviations, punctuation differences, transliteration, partial or reordered addresses, missing address values, and a small number of missing names in Sources 2 and 3.

Unicode-safe normalization was implemented using NFKC normalization, case folding, punctuation and symbol cleanup, and whitespace normalization while preserving non-Latin scripts and accents.

### 2.2 Solution Strategy

**Approach Type:** Blocking + supervised pairwise matching

**Core Innovation:** A scalable disk-backed V2 blocking layer combined with leakage-safe pairwise features, deterministic negative sampling, hard-negative mining, and tree-based classification.

The overall pipeline is:

```text
Raw business records
        ↓
Unicode-safe normalization
        ↓
Token statistics
        ↓
V2 candidate blocking
        ↓
Candidate pairs
        ↓
13 pairwise matching features
        ↓
Supervised matching model
        ↓
Predicted matches

3. Candidate Generation (Blocking)

The candidate-generation stage reduces the full comparison space before pairwise matching.

Blocking keys used

The frozen V2 blocker uses:

Exact normalized business name
Exact compact business name
Rare normalized-name tokens
Top-two globally rarest name tokens
Rare normalized-address tokens
Top-two globally rarest address tokens
Country filtering

The validated rare-token frequency threshold is 10,000.

Scalable candidate generation

The candidate generator uses a disk-backed SQLite/FTS5 index rather than maintaining the full candidate structure in Python memory.

The S2 + S3 index contains approximately 10.32 million candidate records.

The generator supports:

index rebuilding
limiting Source 1 rows
selecting exact Source 1 IDs
configurable output paths
configurable index paths
Candidate benchmark

On the 1,000-S1 benchmark sample:

Candidate links: 8,631,707
Pair-level candidate recall: 98.3193%
Average entity recall: 98.2813%
Matched entities with 100% captured matches: 888 of 934

The V2 blocker was selected after comparing broader blocking alternatives with substantially larger candidate volumes.

Candidate recall strategy

Multiple blocking channels were combined rather than relying on a single exact key. Candidate recall was measured directly against the training ground truth on the benchmark sample.

4. Matching Model
Features used

The matching stage uses 13 pairwise features.

Name features

Normalized-name exact agreement
Compact-name exact agreement
Name token Jaccard similarity
Name character similarity

Address features

Address token Jaccard similarity
Address character similarity

Other features

Number overlap
Number Jaccard similarity
Country agreement
Source 1 name-missing indicator
Candidate name-missing indicator
Source 1 address-missing indicator
Candidate address-missing indicator
Training methodology

The pair dataset uses a grouped Source 1-level train/validation split to reduce leakage across the validation boundary.

Deterministic negative sampling was used for the initial training data. Hard-negative mining was then used to construct a second leakage-safe training dataset.

The initial benchmark feature dataset contains:

21,348 rows
3,393 positive pairs
17,955 negative pairs

The hard-negative training dataset contains 22,192 rows after leakage checks.

Logistic Regression

A Logistic Regression baseline was trained using the 13 pairwise features.

A second Logistic V2 model was trained after hard-negative mining.

On the 200-S1 validation holdout, Logistic V2 at threshold 0.870 achieved:

Precision: 0.9185
Recall: 0.6485
Pair F0.5: 0.8479
Macro F0.5: 0.7592
Tree V3

A HistGradientBoostingClassifier was trained using the same 13 features and leakage-safe split.

On the same full validation candidate set of 1,671,638 links, threshold 0.950 produced:

Precision: 0.9231
Recall: 0.7192
Pair F0.5: 0.8736
Macro F0.5: 0.8145
TP / FP / FN: 456 / 38 / 178
5. Results & Error Analysis
Validation results

The models were evaluated on the same 200-S1 validation holdout using the full candidate set.
| Model       | Threshold | Precision | Recall | Pair F0.5 | Macro F0.5 |
| ----------- | --------: | --------: | -----: | --------: | ---------: |
| Logistic V2 |     0.870 |    0.9185 | 0.6485 |    0.8479 |     0.7592 |
| Tree V3     |     0.950 |    0.9231 | 0.7192 |    0.8736 |     0.8145 |
F_0.5 Score (macro): 0.8145 (Tree V3, 200-S1 validation holdout, threshold 0.950)
Common false positives (wrong merges): 38 false positives were observed for Tree V3 at the current validation threshold. The validation report records false-positive analysis but does not specify a single dominant false-positive pattern.
Common false negatives (missed matches): 178 false negatives were observed for Tree V3 at the current validation threshold. For Logistic V2, analysis separated model false negatives from blocker misses; at threshold 0.87 there were 217 model false negatives and 9 blocker misses.
Error analysis

For Logistic V2, errors were separated into:

model false negatives
blocker misses
false positives

At threshold 0.87, the 200-S1 holdout contained 217 model false negatives and 9 blocker misses.

This showed that most of the remaining recall loss for Logistic V2 was associated with the matching model rather than the blocking stage.

Tree V3 was evaluated using the same candidate set and validation split, allowing model performance to be compared under the same evaluation setup.

6. Conclusion

The project implements a scalable entity resolution pipeline covering data preprocessing, token statistics, candidate blocking, scalable candidate generation, pairwise feature engineering, hard-negative training, and tree-based matching.

The V2 blocker achieved approximately 98.32% pair-level recall on the benchmark sample, while the Tree V3 model achieved Macro F0.5 of 0.8145 on the 200-S1 full-candidate validation holdout at threshold 0.950.
Appendix
A. Code Artefacts

The main implementation is maintained under:

code/business_entity_resolution/src/

Important components include:

preprocessing.py
similarity.py
generate_candidate_pairs.py
validate_candidate_pairs.py
build_pair_features.py
mine_hard_negatives.py
build_hard_negative_training.py
train_hard_negative_model.py
train_tree_model.py
evaluate_tree_model.py
analyze_v2_errors.py
summarize_v2_errors.py
sweep_full_candidate_thresholds.py

The project separates source code from generated datasets, indexes, models, and evaluation outputs.

B. Additional Results

Verified development statistics include:

Development Source 1: 2,206,821 rows
Development Source 2: 5,034,616 rows
Development Source 3: 5,285,603 rows
Ground Truth: 2,206,821 rows
V2 benchmark candidate links: 8,631,707
V2 benchmark pair recall: 98.3193%
Benchmark feature rows: 21,348
Hard-negative training rows: 22,192
Tree V3 validation candidate links: 1,671,638