# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** TriForge 
**Team Members:** A R Yashaswi, Jessa Mariya Joe, Sachita R  
**Submission Date:** 27-09-2026

---

## 1. Executive Summary
We developed a scalable business entity resolution pipeline for identifying corresponding business records across Source 1, Source 2, and Source 3. The solution combines Unicode-safe preprocessing, token-frequency-based blocking, disk-backed SQLite/FTS5 candidate generation, pairwise feature engineering, leakage-safe supervised learning, hard-negative mining, and a tree-based matching model.

The final development model, Tree V3, achieved a Macro F0.5 of 0.8167 on a 200-S1 validation holdout using the full candidate set at a threshold of 0.951.

---

## 2. Methodology

### 2.1 Problem Analysis
The task contains three independent business-record sources without a universal shared identifier. Source 1 acts as the reference source, while Source 2 and Source 3 contain records that may correspond to Source 1 entities.

A Source 1 entity may correspond to zero, one, or multiple records from Source 2 and Source 3.

The development data contains:

Source 1: 2,206,821 rows
Source 2: 5,034,616 rows
Source 3: 5,285,603 rows
Ground Truth: 2,206,821 Source 1 rows

Observed data issues included inconsistent business names, abbreviations, punctuation differences, transliteration, partial or reordered addresses, missing address values, and some missing names.

Unicode-safe normalization was implemented using NFKC normalization, case folding, punctuation and symbol cleanup, and whitespace normalization while preserving non-Latin scripts and accented characters

### 2.2 Solution Strategy
Approach Type: Blocking + supervised pairwise matching

Core Innovation: A scalable disk-backed V2 blocking layer combined with leakage-safe pairwise feature engineering, deterministic negative sampling, hard-negative mining, and tree-based classification.

The overall development pipeline is:

Raw business records -> Unicode-safe normalization -> Token-frequency statistics -> V2 candidate blocking -> Candidate pairs -> 13 pairwise matching features -> Leakage-safe supervised training -> Tree-based matching model -> Threshold-based match decision

---

## 3. Candidate Generation (Blocking)
The candidate-generation stage reduces the comparison space before pairwise matching.

Blocking keys used:

The frozen V2 blocker uses:

Exact normalized business name
Exact compact business name
Rare normalized-name tokens
Top-two globally rarest name tokens
Rare normalized-address tokens
Top-two globally rarest address tokens
Country filtering

The validated single-token frequency threshold was 10,000.

Scalable candidate generation:

The candidate generator uses a disk-backed SQLite/FTS5 index instead of maintaining the complete candidate structure in Python memory. This allows the large Source 2 and Source 3 datasets to be indexed once and queried for Source 1 entities.

The development/test SQLite blocking index contains approximately 10 million Source 2 and Source 3 records.

The generator supports:

index rebuilding
Source 1 row limits
exact Source 1 ID selection
configurable source/index/output paths
deterministic candidate output

Candidate benchmark:

On the 1,000-S1 benchmark sample:

Candidate links: 8,631,707
Pair-level candidate recall: 98.3193%
Entity-level candidate recall: 95.0749%
Entities with 100% of ground-truth matches captured: 888 of 934

Multiple blocking channels were combined to reduce the risk of losing true matches through reliance on a single key. Candidate recall was measured directly against the training ground truth.

---

## 4. Matching Model

Features used:

The matching stage uses 13 pairwise features.

Name features:

Normalized-name exact agreement
Compact-name exact agreement
Name token Jaccard similarity
Name character similarity

Address features:

Address token Jaccard similarity
Address character similarity

Other features:

Number overlap
Number Jaccard similarity
Country agreement
Source 1 name-missing indicator
Candidate name-missing indicator
Source 1 address-missing indicator
Candidate address-missing indicator

Model type:

The development pipeline evaluated Logistic Regression and a tree-based classifier.
A HistGradientBoostingClassifier was selected as Tree V3 using the same 13 pairwise features.

Training methodology:

The pair dataset uses a grouped Source 1-level train/validation split to reduce leakage across the validation boundary.

Deterministic negative sampling was used for the initial training data. Hard-negative mining was then used to construct a second leakage-safe training dataset.

The initial benchmark feature dataset contained:

21,348 rows
3,393 positive pairs
17,955 negative pairs

The hard-negative feature dataset contained:

22,192 rows
800 Source 1 entities for training
200 Source 1 entities for validation
S1-level leakage check: PASS

Threshold selection:

The decision threshold was selected through a validation sweep using Macro F0.5 as the primary evaluation metric.

The final Tree V3 validation threshold was:

0.951

---

## 5. Results & Error Analysis

Validation results:

The development models were evaluated on the same 200-S1 validation holdout using the full candidate set.

| Model       | Threshold |  Precision |     Recall |  Pair F0.5 | Macro F0.5 |
| ----------- | --------: | ---------: | ---------: | ---------: | ---------: |
| Logistic V2 |     0.870 |     0.9185 |     0.6485 |     0.8479 |     0.7592 |
| Tree V3     | **0.951** | **0.9268** | **0.7192** | **0.8762** | **0.8167** |

F0.5 Score (macro): 0.8167

This was obtained by Tree V3 on the 200-S1 validation holdout at threshold 0.951.

False positives:

At the selected Tree V3 threshold:

True positives: 456
False positives: 36
False negatives: 178

The false-positive analysis showed that incorrect matches can occur when several address and name similarity signals are simultaneously high even when the records refer to different businesses.

False negatives:

The remaining false negatives generally occur when a true match has weak or partial name agreement, incomplete address agreement, or missing/low-quality identifying information.

For the earlier Logistic V2 analysis at threshold 0.87, the errors were separated into model false negatives and blocker misses:

Model false negatives: 217
Blocker misses: 9

This indicated that the majority of remaining recall loss for Logistic V2 occurred during the matching stage rather than candidate blocking.

Tree V3 was evaluated using the same validation split and full candidate set, allowing direct comparison under the same evaluation setup.

Candidate-generation versus final submission pipeline:

The full V2 candidate configuration was used for development validation and model evaluation. During final test execution, candidate volume and runtime became the main computational constraint, so a compact candidate-generation strategy was used for the submission pipeline. The validation score reported above therefore represents the development model evaluation, not the final test-set score.

---

## 6. Conclusion
The project implements a scalable business entity resolution pipeline covering preprocessing, token statistics, candidate blocking, scalable candidate generation, pairwise feature engineering, hard-negative training, and tree-based matching.

The V2 blocker achieved 98.3193% pair-level candidate recall on the benchmark sample, while Tree V3 achieved a Macro F0.5 of 0.8167 on the 200-S1 validation holdout at threshold 0.951.

---

## Appendix

### A. Code Artefacts
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

The final inference and submission utilities additionally include:
final_inference.py
complete_submission.py

### B. Additional Results
Verified development statistics include:

Development Source 1: 2,206,821 rows
Development Source 2: 5,034,616 rows
Development Source 3: 5,285,603 rows
Ground Truth: 2,206,821 rows
V2 benchmark candidate links: 8,631,707
V2 benchmark pair recall: 98.3193%
V2 benchmark entity recall: 95.0749%
Benchmark feature rows: 21,348
Hard-negative feature rows: 22,192
Tree V3 validation candidate links: 1,671,638
Tree V3 validation Precision: 0.9268
Tree V3 validation Recall: 0.7192
Tree V3 validation Pair F0.5: 0.8762
Tree V3 validation Macro F0.5: 0.8167
Selected Tree V3 threshold: 0.951

---
