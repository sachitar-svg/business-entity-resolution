from pathlib import Path
import random

import pandas as pd

from build_pair_features import (
    calculate_features,
    load_selected_candidate_records,
    load_selected_s1_records,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_FEATURE_PATH = (
    PROJECT_ROOT
    / "output"
    / "benchmark_pair_features.parquet"
)

HARD_NEGATIVE_PATH = (
    PROJECT_ROOT
    / "output"
    / "mined_hard_negatives.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "hard_negative_pair_features.parquet"
)

# New validation size.
NEW_VALIDATION_S1_COUNT = 200

# Separate seed from the original split.
RANDOM_STATE = 2026


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "name_exact",
    "name_compact_exact",
    "name_token_jaccard",
    "name_char_similarity",
    "address_token_jaccard",
    "address_char_similarity",
    "number_overlap",
    "number_jaccard",
    "country_match",
    "s1_name_missing",
    "candidate_name_missing",
    "s1_address_missing",
    "candidate_address_missing",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING LEAKAGE-SAFE HARD-NEGATIVE TRAINING DATASET")
    print("=" * 70)

    print(
        f"Base features       : {BASE_FEATURE_PATH}"
    )

    print(
        f"Hard negatives      : {HARD_NEGATIVE_PATH}"
    )

    print(
        f"Output              : {OUTPUT_PATH}"
    )

    print(
        f"New validation S1   : {NEW_VALIDATION_S1_COUNT}"
    )

    print(
        f"Random state        : {RANDOM_STATE}"
    )

    # --------------------------------------------------------
    # Load base feature dataset
    # --------------------------------------------------------

    print(
        "\nLoading base feature dataset..."
    )

    base_df = pd.read_parquet(
        BASE_FEATURE_PATH
    )

    print(
        f"Base feature rows: "
        f"{len(base_df):,}"
    )

    # --------------------------------------------------------
    # Identify original train / validation S1 groups
    # --------------------------------------------------------

    original_validation_s1 = set(
        base_df.loc[
            base_df["split"] == "validation",
            "s1_id",
        ]
        .astype(str)
    )

    original_train_s1 = set(
        base_df.loc[
            base_df["split"] == "train",
            "s1_id",
        ]
        .astype(str)
    )

    print(
        "\nOriginal split:"
    )

    print(
        f"  Original train S1      : "
        f"{len(original_train_s1):,}"
    )

    print(
        f"  Original validation S1 : "
        f"{len(original_validation_s1):,}"
    )

    # --------------------------------------------------------
    # Create NEW validation group
    # --------------------------------------------------------

    if NEW_VALIDATION_S1_COUNT >= len(
        original_train_s1
    ):
        raise ValueError(
            "New validation size must be smaller "
            "than the original training S1 count."
        )

    rng = random.Random(
        RANDOM_STATE
    )

    new_validation_s1 = set(
        rng.sample(
            sorted(original_train_s1),
            NEW_VALIDATION_S1_COUNT,
        )
    )

    new_training_s1 = (
        original_train_s1
        - new_validation_s1
        | original_validation_s1
    )

    print(
        "\nNEW grouped split:"
    )

    print(
        f"  New training S1       : "
        f"{len(new_training_s1):,}"
    )

    print(
        f"  New validation S1     : "
        f"{len(new_validation_s1):,}"
    )

    # --------------------------------------------------------
    # Reassign base feature rows
    # --------------------------------------------------------

    base_df["split"] = base_df["s1_id"].astype(str).map(
        lambda s1_id: (
            "validation"
            if s1_id in new_validation_s1
            else "train"
        )
    )

    # --------------------------------------------------------
    # Load mined hard negatives
    # --------------------------------------------------------

    print(
        "\nLoading mined hard negatives..."
    )

    hard_negative_df = pd.read_parquet(
        HARD_NEGATIVE_PATH
    )

    print(
        f"Hard-negative rows: "
        f"{len(hard_negative_df):,}"
    )

    if hard_negative_df.empty:
        raise ValueError(
            "Hard-negative dataset is empty."
        )

    required_hard_columns = {
        "s1_id",
        "candidate_id",
        "label",
    }

    missing_columns = (
        required_hard_columns
        - set(hard_negative_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Hard-negative dataset is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # --------------------------------------------------------
    # Safety check:
    # hard negatives must NEVER enter validation.
    # --------------------------------------------------------

    hard_negative_s1 = set(
        hard_negative_df["s1_id"]
        .astype(str)
    )

    leakage_s1 = (
        hard_negative_s1
        & new_validation_s1
    )

    if leakage_s1:
        raise RuntimeError(
            "LEAKAGE DETECTED: hard-negative S1 entities "
            "overlap with new validation S1 entities.\n"
            f"Overlapping entities: {len(leakage_s1):,}"
        )

    print(
        "Hard-negative leakage check: PASS"
    )

    # --------------------------------------------------------
    # Remove hard-negative pairs already present
    # in the base feature dataset.
    # --------------------------------------------------------

    base_pair_keys = set(
        zip(
            base_df["s1_id"].astype(str),
            base_df["candidate_id"].astype(str),
        )
    )

    hard_negative_df["s1_id"] = (
        hard_negative_df["s1_id"]
        .astype(str)
    )

    hard_negative_df["candidate_id"] = (
        hard_negative_df["candidate_id"]
        .astype(str)
    )

    hard_negative_df["label"] = 0

    hard_negative_keys = list(
        zip(
            hard_negative_df["s1_id"],
            hard_negative_df["candidate_id"],
        )
    )

    new_hard_negative_mask = [
        key not in base_pair_keys
        for key in hard_negative_keys
    ]

    hard_negative_df = (
        hard_negative_df.loc[
            new_hard_negative_mask
        ]
        .copy()
    )

    print(
        f"New hard-negative rows after "
        f"deduplication: {len(hard_negative_df):,}"
    )

    # --------------------------------------------------------
    # Load records required for hard-negative features
    # --------------------------------------------------------

    selected_s1_ids = set(
        hard_negative_df["s1_id"]
    )

    selected_candidate_ids = set(
        hard_negative_df["candidate_id"]
    )

    print(
        "\nLoading S1 records required for hard negatives..."
    )

    s1_records = load_selected_s1_records(
        selected_s1_ids
    )

    print(
        f"S1 records loaded: "
        f"{len(s1_records):,}"
    )

    print(
        "\nLoading candidate records required for hard negatives..."
    )

    candidate_records = (
        load_selected_candidate_records(
            selected_candidate_ids
        )
    )

    print(
        f"Candidate records loaded: "
        f"{len(candidate_records):,}"
    )

    # --------------------------------------------------------
    # Build hard-negative feature rows
    # --------------------------------------------------------

    print(
        "\nCalculating hard-negative features..."
    )

    hard_feature_rows = []

    missing_s1 = 0
    missing_candidates = 0

    for index, row in enumerate(
        hard_negative_df.itertuples(
            index=False
        ),
        start=1,
    ):

        s1_id = str(
            row.s1_id
        )

        candidate_id = str(
            row.candidate_id
        )

        s1_record = s1_records.get(
            s1_id
        )

        candidate_record = (
            candidate_records.get(
                candidate_id
            )
        )

        if s1_record is None:
            missing_s1 += 1
            continue

        if candidate_record is None:
            missing_candidates += 1
            continue

        hard_feature_rows.append(
            calculate_features(
                s1_record,
                candidate_record,
                0,
                "train",
            )
        )

        if index % 100 == 0:
            print(
                f"Hard-negative features built: "
                f"{index:,}/"
                f"{len(hard_negative_df):,}",
                flush=True,
            )

    hard_features_df = pd.DataFrame(
        hard_feature_rows
    )

    print()
    print(
        f"Missing S1 records       : "
        f"{missing_s1:,}"
    )

    print(
        f"Missing candidate records: "
        f"{missing_candidates:,}"
    )

    print(
        f"Hard feature rows built  : "
        f"{len(hard_features_df):,}"
    )

    if missing_s1 or missing_candidates:
        raise RuntimeError(
            "Some hard-negative records could not be resolved."
        )

    # --------------------------------------------------------
    # Ensure hard negatives are training rows
    # --------------------------------------------------------

    hard_features_df["split"] = "train"

    # --------------------------------------------------------
    # Combine base + hard negatives
    # --------------------------------------------------------

    combined_df = pd.concat(
        [
            base_df,
            hard_features_df,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Final duplicate check
    # --------------------------------------------------------

    before_dedup = len(
        combined_df
    )

    combined_df = (
        combined_df
        .drop_duplicates(
            subset=[
                "s1_id",
                "candidate_id",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    duplicate_count = (
        before_dedup
        - len(combined_df)
    )

    print()
    print(
        f"Duplicate pair rows removed: "
        f"{duplicate_count:,}"
    )

    # --------------------------------------------------------
    # Final split validation
    # --------------------------------------------------------

    validation_rows = combined_df[
        combined_df["split"] == "validation"
    ]

    train_rows = combined_df[
        combined_df["split"] == "train"
    ]

    validation_s1_check = set(
        validation_rows["s1_id"].astype(str)
    )

    train_s1_check = set(
        train_rows["s1_id"].astype(str)
    )

    overlap = (
        validation_s1_check
        & train_s1_check
    )

    if overlap:
        raise RuntimeError(
            "S1 leakage detected after combining datasets."
        )

    print(
        "Final grouped split leakage check: PASS"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    expected_columns = [
        "s1_id",
        "candidate_id",
        "label",
        "split",
    ] + FEATURE_COLUMNS

    missing_final = [
        column
        for column in expected_columns
        if column not in combined_df.columns
    ]

    if missing_final:
        raise ValueError(
            "Final dataset is missing columns: "
            + ", ".join(missing_final)
        )

    # Keep expected column order.
    combined_df = combined_df[
        expected_columns
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    train_positive = int(
        train_rows["label"].sum()
    )

    train_negative = int(
        (train_rows["label"] == 0).sum()
    )

    valid_positive = int(
        validation_rows["label"].sum()
    )

    valid_negative = int(
        (validation_rows["label"] == 0).sum()
    )

    print()
    print("=" * 70)
    print("HARD-NEGATIVE TRAINING DATASET COMPLETE")
    print("=" * 70)

    print(
        f"Total feature rows       : "
        f"{len(combined_df):,}"
    )

    print(
        f"Hard negatives added     : "
        f"{len(hard_features_df):,}"
    )

    print()
    print(
        "Training:"
    )

    print(
        f"  S1 entities            : "
        f"{len(train_s1_check):,}"
    )

    print(
        f"  Rows                   : "
        f"{len(train_rows):,}"
    )

    print(
        f"  Positive rows          : "
        f"{train_positive:,}"
    )

    print(
        f"  Negative rows          : "
        f"{train_negative:,}"
    )

    print()
    print(
        "Validation:"
    )

    print(
        f"  S1 entities            : "
        f"{len(validation_s1_check):,}"
    )

    print(
        f"  Rows                   : "
        f"{len(validation_rows):,}"
    )

    print(
        f"  Positive rows          : "
        f"{valid_positive:,}"
    )

    print(
        f"  Negative rows          : "
        f"{valid_negative:,}"
    )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()