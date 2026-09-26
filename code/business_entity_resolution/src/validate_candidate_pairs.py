from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "output"
    / "candidate_pairs_benchmark_sample.tsv"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)


# ============================================================
# LOAD CANDIDATE FILE
# ============================================================

print("=" * 70)
print("CANDIDATE PAIR VALIDATION")
print("=" * 70)

print("\nLoading candidate file...")

candidates_df = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

print(
    f"Candidate rows: "
    f"{len(candidates_df):,}"
)


# ============================================================
# BASIC STRUCTURE CHECK
# ============================================================

expected_columns = {
    "source1_entity_id",
    "candidate_entity_ids",
}

actual_columns = set(
    candidates_df.columns
)

if not expected_columns.issubset(
    actual_columns
):

    raise ValueError(
        "Candidate file does not have the "
        "required columns:\n"
        f"Expected: {expected_columns}\n"
        f"Found: {actual_columns}"
    )


duplicate_s1 = (
    candidates_df["source1_entity_id"]
    .duplicated()
    .sum()
)

print(
    f"Duplicate S1 rows: "
    f"{duplicate_s1:,}"
)


# ============================================================
# CANDIDATE COUNT STATISTICS
# ============================================================

candidate_counts = (
    candidates_df["candidate_entity_ids"]
    .apply(
        lambda x:
        0
        if not x
        else len(
            set(
                item.strip()
                for item in x.split(",")
                if item.strip()
            )
        )
    )
)

print("\n" + "=" * 70)
print("CANDIDATE VOLUME")
print("=" * 70)

print(
    f"Mean        : "
    f"{candidate_counts.mean():,.2f}"
)

print(
    f"Median      : "
    f"{candidate_counts.median():,.2f}"
)

print(
    f"95th pct    : "
    f"{np.percentile(candidate_counts, 95):,.2f}"
)

print(
    f"Maximum     : "
    f"{candidate_counts.max():,.0f}"
)

print(
    f"Zero-candidate S1: "
    f"{(candidate_counts == 0).sum():,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt_df = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

print(
    f"Ground-truth rows: "
    f"{len(gt_df):,}"
)


# ============================================================
# RESTRICT TO THE SAME 1,000 S1 ENTITIES
# ============================================================

sample_ids = set(
    candidates_df[
        "source1_entity_id"
    ]
)

gt_sample = gt_df[
    gt_df["source1_entity_id"].isin(
        sample_ids
    )
].copy()


print(
    f"Ground-truth rows for test sample: "
    f"{len(gt_sample):,}"
)


# ============================================================
# BUILD GROUND-TRUTH LOOKUP
# ============================================================

ground_truth = {}

for row in gt_sample.itertuples(
    index=False
):

    s1_id = row.source1_entity_id

    raw_matches = (
        row.matched_entity_ids
    )

    if not raw_matches:

        ground_truth[s1_id] = set()

    else:

        ground_truth[s1_id] = {
            item.strip()
            for item in raw_matches.split(",")
            if item.strip()
        }


# ============================================================
# CALCULATE BLOCKING RECALL
# ============================================================

per_entity_recalls = []

total_true_matches = 0
total_captured_matches = 0

entities_with_true_matches = 0
entities_full_recall = 0

entities_with_missed_matches = 0


for row in candidates_df.itertuples(
    index=False
):

    s1_id = row.source1_entity_id

    true_matches = ground_truth.get(
        s1_id,
        set()
    )

    candidate_ids = (
        {
            item.strip()
            for item in row.candidate_entity_ids.split(",")
            if item.strip()
        }
        if row.candidate_entity_ids
        else set()
    )


    true_count = len(
        true_matches
    )

    captured = (
        true_matches
        & candidate_ids
    )

    captured_count = len(
        captured
    )


    if true_count > 0:

        entities_with_true_matches += 1

        recall = (
            captured_count
            / true_count
        )

        per_entity_recalls.append(
            recall
        )

        if captured_count == true_count:

            entities_full_recall += 1

        else:

            entities_with_missed_matches += 1


    total_true_matches += true_count

    total_captured_matches += captured_count


# ============================================================
# SUMMARY
# ============================================================

average_recall = (
    np.mean(per_entity_recalls)
    if per_entity_recalls
    else 0.0
)

pair_recall = (
    total_captured_matches
    / total_true_matches
    if total_true_matches
    else 0.0
)


print("\n" + "=" * 70)
print("BLOCKING RECALL")
print("=" * 70)

print(
    f"Entities evaluated        : "
    f"{len(candidates_df):,}"
)

print(
    f"Entities with true matches: "
    f"{entities_with_true_matches:,}"
)

print(
    f"Total true matches        : "
    f"{total_true_matches:,}"
)

print(
    f"True matches captured     : "
    f"{total_captured_matches:,}"
)

print(
    f"Pair-level recall         : "
    f"{pair_recall:.4%}"
)

print(
    f"Average entity recall     : "
    f"{average_recall:.4%}"
)

print(
    f"100% recall entities      : "
    f"{entities_full_recall:,}"
)

print(
    f"Entities with misses      : "
    f"{entities_with_missed_matches:,}"
)


# ============================================================
# SHOW MISSED ENTITIES
# ============================================================

print("\n" + "=" * 70)
print("MISSED TRUE-MATCH ENTITIES")
print("=" * 70)

missed_rows = []


for row in candidates_df.itertuples(
    index=False
):

    s1_id = row.source1_entity_id

    true_matches = ground_truth.get(
        s1_id,
        set()
    )

    if not true_matches:
        continue

    candidate_ids = (
        {
            item.strip()
            for item in row.candidate_entity_ids.split(",")
            if item.strip()
        }
        if row.candidate_entity_ids
        else set()
    )

    missed = (
        true_matches
        - candidate_ids
    )

    if missed:

        missed_rows.append(
            {
                "s1_id": s1_id,
                "true_count": len(true_matches),
                "captured_count": len(
                    true_matches & candidate_ids
                ),
                "missed_count": len(missed),
                "missed_ids": ",".join(
                    sorted(missed)
                ),
            }
        )


if missed_rows:

    missed_df = pd.DataFrame(
        missed_rows
    )

    print(
        missed_df.to_string(
            index=False
        )
    )

else:

    print(
        "No missed true matches."
    )


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

if duplicate_s1 != 0:

    print(
        "WARNING: duplicate Source-1 rows found."
    )

if average_recall >= 0.97:

    print(
        "Candidate recall is in the expected range."
    )

else:

    print(
        "WARNING: candidate recall is below the "
        "expected V2 range."
    )

print(
    "\nDone."
)