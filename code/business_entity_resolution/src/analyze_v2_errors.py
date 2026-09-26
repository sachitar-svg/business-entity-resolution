from pathlib import Path
import csv
import sys

import joblib
import numpy as np
import pandas as pd

from build_pair_features import (
    calculate_features,
    load_ground_truth,
    load_selected_candidate_records,
    load_selected_s1_records,
    safe_text,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_PATH = (
    PROJECT_ROOT
    / "output"
    / "hard_negative_pair_features.parquet"
)

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

MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "hard_negative_logistic_model.joblib"
)

ENTITY_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "v2_error_entity_summary.csv"
)

FP_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "v2_false_positives.csv"
)

MODEL_FN_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "v2_model_false_negatives.csv"
)

BLOCKER_MISS_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "v2_blocker_misses.csv"
)

THRESHOLD = 0.87

csv.field_size_limit(sys.maxsize)


# ============================================================
# FEATURES
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
# LOAD CANDIDATES
# ============================================================

def load_validation_candidates(
    candidate_path,
    validation_s1_ids,
):
    """
    Load all candidates for the selected validation S1 entities.
    """

    candidate_map = {}

    with open(
        candidate_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        header = file.readline().strip().split("\t")

        if len(header) != 2:
            raise ValueError(
                "Candidate file must contain exactly two columns."
            )

        if header[0] != "source1_entity_id":
            raise ValueError(
                "Expected first column: source1_entity_id"
            )

        if header[1] != "candidate_entity_ids":
            raise ValueError(
                "Expected second column: candidate_entity_ids"
            )

        for line_number, line in enumerate(
            file,
            start=2,
        ):

            line = line.rstrip("\r\n")

            if not line:
                continue

            parts = line.split("\t", 1)

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid candidate row at line {line_number}."
                )

            s1_id = safe_text(parts[0])

            if s1_id not in validation_s1_ids:
                continue

            candidate_text = safe_text(parts[1])

            if candidate_text:
                candidate_ids = [
                    value.strip()
                    for value in candidate_text.split(",")
                    if value.strip()
                ]
            else:
                candidate_ids = []

            candidate_map[s1_id] = candidate_ids

    return candidate_map


# ============================================================
# ENTITY F0.5
# ============================================================

def entity_f05(
    true_ids,
    predicted_ids,
):
    true_ids = set(true_ids)
    predicted_ids = set(predicted_ids)

    tp = len(true_ids & predicted_ids)
    fp = len(predicted_ids - true_ids)
    fn = len(true_ids - predicted_ids)

    denominator = (
        1.25 * tp
        + 0.25 * fn
        + fp
    )

    if denominator == 0:
        return 1.0, tp, fp, fn

    score = (
        1.25 * tp
        / denominator
    )

    return score, tp, fp, fn


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V2 ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"Model              : {MODEL_PATH}"
    )

    print(
        f"Threshold          : {THRESHOLD:.2f}"
    )

    print(
        f"Candidate file     : {CANDIDATE_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading V2 model...")

    model_bundle = joblib.load(
        MODEL_PATH
    )

    model = model_bundle["model"]

    feature_columns = model_bundle[
        "feature_columns"
    ]

    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature columns do not match expected columns."
        )

    print(
        f"Feature columns: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Validation S1
    # --------------------------------------------------------

    print(
        "\nLoading validation S1 IDs..."
    )

    split_df = pd.read_parquet(
        FEATURE_PATH,
        columns=[
            "s1_id",
            "split",
        ],
    )

    validation_s1_ids = set(
        split_df.loc[
            split_df["split"] == "validation",
            "s1_id",
        ]
        .astype(str)
        .tolist()
    )

    del split_df

    print(
        f"Validation S1 entities: "
        f"{len(validation_s1_ids):,}"
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print(
        "\nLoading ground truth..."
    )

    ground_truth = load_ground_truth(
        GROUND_TRUTH_PATH
    )

    print(
        f"Ground-truth S1 rows: "
        f"{len(ground_truth):,}"
    )

    # --------------------------------------------------------
    # Candidate lists
    # --------------------------------------------------------

    print(
        "\nLoading candidate lists..."
    )

    candidate_map = load_validation_candidates(
        CANDIDATE_PATH,
        validation_s1_ids,
    )

    total_candidate_links = sum(
        len(ids)
        for ids in candidate_map.values()
    )

    unique_candidate_ids = {
        candidate_id
        for ids in candidate_map.values()
        for candidate_id in ids
    }

    print(
        f"Candidate S1 rows: "
        f"{len(candidate_map):,}"
    )

    print(
        f"Candidate links: "
        f"{total_candidate_links:,}"
    )

    print(
        f"Unique candidate IDs: "
        f"{len(unique_candidate_ids):,}"
    )

    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    print(
        "\nLoading S1 records..."
    )

    s1_records = load_selected_s1_records(
        validation_s1_ids
    )

    print(
        f"S1 records loaded: "
        f"{len(s1_records):,}"
    )

    print(
        "\nLoading candidate records..."
    )

    candidate_records = (
        load_selected_candidate_records(
            unique_candidate_ids
        )
    )

    print(
        f"Candidate records loaded: "
        f"{len(candidate_records):,}"
    )

    # --------------------------------------------------------
    # Results storage
    # --------------------------------------------------------

    entity_results = []

    false_positive_rows = []
    model_false_negative_rows = []
    blocker_miss_rows = []

    total_tp = 0
    total_fp = 0
    total_model_fn = 0
    total_blocker_fn = 0

    # --------------------------------------------------------
    # Process each S1
    # --------------------------------------------------------

    print(
        "\nAnalyzing validation entities..."
    )

    for index, s1_id in enumerate(
        sorted(validation_s1_ids),
        start=1,
    ):

        s1_record = s1_records.get(
            s1_id
        )

        if s1_record is None:
            print(
                f"WARNING: missing S1 record: {s1_id}"
            )
            continue

        candidate_ids = candidate_map.get(
            s1_id,
            [],
        )

        # Remove accidental duplicates while preserving order.
        candidate_ids = list(
            dict.fromkeys(candidate_ids)
        )

        true_ids = set(
            ground_truth.get(
                s1_id,
                set(),
            )
        )

        candidate_set = set(
            candidate_ids
        )

        # ----------------------------------------------------
        # Blocker misses
        # ----------------------------------------------------

        blocker_missed_ids = (
            true_ids
            - candidate_set
        )

        for candidate_id in sorted(
            blocker_missed_ids
        ):

            blocker_miss_rows.append(
                {
                    "s1_id": s1_id,
                    "candidate_id": candidate_id,
                    "error_type": "blocker_miss",
                }
            )

        # ----------------------------------------------------
        # Build features for candidates
        # ----------------------------------------------------

        feature_rows = []
        usable_candidate_ids = []

        for candidate_id in candidate_ids:

            candidate_record = (
                candidate_records.get(
                    candidate_id
                )
            )

            if candidate_record is None:
                continue

            feature_rows.append(
                calculate_features(
                    s1_record,
                    candidate_record,
                    0,
                    "error_analysis",
                )
            )

            usable_candidate_ids.append(
                candidate_id
            )

        # ----------------------------------------------------
        # Score candidates
        # ----------------------------------------------------

        scored_pairs = []

        if feature_rows:

            feature_df = pd.DataFrame(
                feature_rows
            )

            X = (
                feature_df[
                    feature_columns
                ]
                .astype(float)
            )

            probabilities = (
                model.predict_proba(X)[:, 1]
            )

            for row_number, (
                candidate_id,
                probability,
            ) in enumerate(
                zip(
                    usable_candidate_ids,
                    probabilities,
                )
            ):

                scored_pairs.append(
                    {
                        "candidate_id": candidate_id,
                        "probability": float(
                            probability
                        ),
                        "features": feature_rows[
                            row_number
                        ],
                    }
                )

        # ----------------------------------------------------
        # Predicted matches
        # ----------------------------------------------------

        predicted_ids = {
            row["candidate_id"]
            for row in scored_pairs
            if row["probability"] >= THRESHOLD
        }

        # ----------------------------------------------------
        # True positives
        # ----------------------------------------------------

        true_positive_ids = (
            true_ids
            & predicted_ids
        )

        # ----------------------------------------------------
        # Model false negatives
        #
        # These are true matches that WERE in the candidate
        # set but received probability below threshold.
        # ----------------------------------------------------

        model_false_negative_ids = (
            true_ids
            & candidate_set
            - predicted_ids
        )

        for candidate_id in sorted(
            model_false_negative_ids
        ):

            probability = None
            feature_values = {}

            for row in scored_pairs:

                if row["candidate_id"] == candidate_id:

                    probability = row[
                        "probability"
                    ]

                    feature_values = row[
                        "features"
                    ]

                    break

            model_false_negative_rows.append(
                {
                    "s1_id": s1_id,
                    "candidate_id": candidate_id,
                    "model_probability": probability,
                    **{
                        key: feature_values.get(
                            key
                        )
                        for key in FEATURE_COLUMNS
                    },
                }
            )

        # ----------------------------------------------------
        # False positives
        # ----------------------------------------------------

        false_positive_ids = (
            predicted_ids
            - true_ids
        )

        for candidate_id in sorted(
            false_positive_ids
        ):

            probability = None
            feature_values = {}

            for row in scored_pairs:

                if row["candidate_id"] == candidate_id:

                    probability = row[
                        "probability"
                    ]

                    feature_values = row[
                        "features"
                    ]

                    break

            false_positive_rows.append(
                {
                    "s1_id": s1_id,
                    "candidate_id": candidate_id,
                    "model_probability": probability,
                    **{
                        key: feature_values.get(
                            key
                        )
                        for key in FEATURE_COLUMNS
                    },
                }
            )

        # ----------------------------------------------------
        # Entity metrics
        # ----------------------------------------------------

        score, tp, fp, fn = entity_f05(
            true_ids,
            predicted_ids,
        )

        model_fn_count = len(
            model_false_negative_ids
        )

        blocker_fn_count = len(
            blocker_missed_ids
        )

        total_tp += tp
        total_fp += fp
        total_model_fn += model_fn_count
        total_blocker_fn += blocker_fn_count

        entity_results.append(
            {
                "s1_id": s1_id,
                "true_match_count": len(
                    true_ids
                ),
                "candidate_count": len(
                    candidate_set
                ),
                "predicted_match_count": len(
                    predicted_ids
                ),
                "true_positive_count": tp,
                "false_positive_count": fp,
                "model_false_negative_count": (
                    model_fn_count
                ),
                "blocker_missed_count": (
                    blocker_fn_count
                ),
                "total_false_negative_count": fn,
                "f05": score,
            }
        )

        if index % 25 == 0:
            print(
                f"Processed S1: "
                f"{index:,}/"
                f"{len(validation_s1_ids):,}",
                flush=True,
            )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    entity_df = pd.DataFrame(
        entity_results
    )

    macro_f05 = float(
        entity_df["f05"].mean()
        if not entity_df.empty
        else 0.0
    )

    total_fn = (
        total_model_fn
        + total_blocker_fn
    )

    precision = (
        total_tp
        / (total_tp + total_fp)
        if total_tp + total_fp
        else 0.0
    )

    recall = (
        total_tp
        / (total_tp + total_fn)
        if total_tp + total_fn
        else 0.0
    )

    pair_denominator = (
        1.25 * total_tp
        + 0.25 * total_fn
        + total_fp
    )

    pair_f05 = (
        1.25 * total_tp
        / pair_denominator
        if pair_denominator
        else 1.0
    )

    # --------------------------------------------------------
    # Save entity summary
    # --------------------------------------------------------

    entity_df = entity_df.sort_values(
        [
            "f05",
            "false_positive_count",
            "total_false_negative_count",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    entity_df.to_csv(
        ENTITY_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Save false positives
    # --------------------------------------------------------

    fp_df = pd.DataFrame(
        false_positive_rows
    )

    if not fp_df.empty:
        fp_df = fp_df.sort_values(
            "model_probability",
            ascending=False,
        )

    fp_df.to_csv(
        FP_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Save model false negatives
    # --------------------------------------------------------

    model_fn_df = pd.DataFrame(
        model_false_negative_rows
    )

    if not model_fn_df.empty:
        model_fn_df = model_fn_df.sort_values(
            "model_probability",
            ascending=False,
        )

    model_fn_df.to_csv(
        MODEL_FN_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Save blocker misses
    # --------------------------------------------------------

    blocker_df = pd.DataFrame(
        blocker_miss_rows
    )

    blocker_df.to_csv(
        BLOCKER_MISS_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V2 ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Validation S1 entities : "
        f"{len(entity_df):,}"
    )

    print(
        f"Candidate links        : "
        f"{total_candidate_links:,}"
    )

    print()
    print(
        f"True positives         : "
        f"{total_tp:,}"
    )

    print(
        f"False positives        : "
        f"{total_fp:,}"
    )

    print(
        f"Model false negatives  : "
        f"{total_model_fn:,}"
    )

    print(
        f"Blocker missed pairs   : "
        f"{total_blocker_fn:,}"
    )

    print(
        f"Total false negatives  : "
        f"{total_fn:,}"
    )

    print()
    print(
        f"Precision              : "
        f"{precision:.4f}"
    )

    print(
        f"Recall                 : "
        f"{recall:.4f}"
    )

    print(
        f"Pair F0.5              : "
        f"{pair_f05:.4f}"
    )

    print(
        f"Macro F0.5             : "
        f"{macro_f05:.4f}"
    )

    print()
    print("Files created:")

    print(
        f"  Entity summary       : {ENTITY_OUTPUT}"
    )

    print(
        f"  False positives      : {FP_OUTPUT}"
    )

    print(
        f"  Model false negatives: {MODEL_FN_OUTPUT}"
    )

    print(
        f"  Blocker misses       : {BLOCKER_MISS_OUTPUT}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()