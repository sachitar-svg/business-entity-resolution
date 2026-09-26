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

RESULT_PATH = (
    PROJECT_ROOT
    / "output"
    / "hard_negative_full_candidate_threshold_results.csv"
)

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    0.995,
    0.999,
]

csv.field_size_limit(sys.maxsize)


# ============================================================
# LOAD CANDIDATES
# ============================================================

def load_validation_candidates(
    candidate_path,
    validation_s1_ids,
):
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

def entity_f05(true_ids, predicted_ids):

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
# SCORE ALL CANDIDATES ONCE
# ============================================================

def score_all_candidates(
    validation_s1_ids,
    candidate_map,
    s1_records,
    candidate_records,
    model,
    feature_columns,
):

    probability_map = {}

    total_links = sum(
        len(ids)
        for ids in candidate_map.values()
    )

    print()
    print(
        f"Total candidate links to score: "
        f"{total_links:,}"
    )

    print()

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

            probability_map[s1_id] = []
            continue

        candidate_ids = candidate_map.get(
            s1_id,
            [],
        )

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
                    "evaluation",
                )
            )

            usable_candidate_ids.append(
                candidate_id
            )

        if not feature_rows:

            probability_map[s1_id] = []

        else:

            feature_df = pd.DataFrame(
                feature_rows
            )

            X = feature_df[
                feature_columns
            ].astype(float)

            probabilities = (
                model.predict_proba(X)[:, 1]
            )

            probability_map[s1_id] = [
                (
                    candidate_id,
                    float(probability),
                )
                for candidate_id, probability
                in zip(
                    usable_candidate_ids,
                    probabilities,
                )
            ]

        if index % 25 == 0:
            print(
                f"Scored validation S1: "
                f"{index:,}/{len(validation_s1_ids):,}",
                flush=True,
            )

    return probability_map


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    threshold,
    validation_s1_ids,
    probability_map,
    ground_truth,
):

    macro_scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    predicted_positive_pairs = 0

    for s1_id in sorted(validation_s1_ids):

        scored_candidates = probability_map.get(
            s1_id,
            [],
        )

        predicted_ids = {
            candidate_id
            for candidate_id, probability
            in scored_candidates
            if probability >= threshold
        }

        true_ids = ground_truth.get(
            s1_id,
            set(),
        )

        score, tp, fp, fn = entity_f05(
            true_ids,
            predicted_ids,
        )

        macro_scores.append(score)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        predicted_positive_pairs += len(
            predicted_ids
        )

    macro_f05 = (
        float(np.mean(macro_scores))
        if macro_scores
        else 0.0
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

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "pair_f05": pair_f05,
        "macro_f05": macro_f05,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "predicted_positive_pairs": (
            predicted_positive_pairs
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HARD-NEGATIVE MODEL FULL-CANDIDATE THRESHOLD SWEEP")
    print("=" * 70)

    print(
        f"Feature file : {FEATURE_PATH}"
    )

    print(
        f"Candidate    : {CANDIDATE_PATH}"
    )

    print(
        f"Model        : {MODEL_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "Loading hard-negative model..."
    )

    model_bundle = joblib.load(
        MODEL_PATH
    )

    model = model_bundle["model"]

    feature_columns = model_bundle[
        "feature_columns"
    ]

    print(
        f"Feature columns: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # New validation S1 IDs
    # --------------------------------------------------------

    print(
        "\nLoading NEW validation S1 IDs..."
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
        f"New validation S1 entities: "
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
        "\nLoading complete candidate lists..."
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
        f"Validation candidate rows: "
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
    # Records
    # --------------------------------------------------------

    print(
        "\nLoading Source-1 records..."
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
    # Score all candidates once
    # --------------------------------------------------------

    print(
        "\nScoring all candidate pairs ONCE..."
    )

    probability_map = score_all_candidates(
        validation_s1_ids,
        candidate_map,
        s1_records,
        candidate_records,
        model,
        feature_columns,
    )

    print(
        "\nAll candidate probabilities generated."
    )

    # --------------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HARD-NEGATIVE MODEL THRESHOLD RESULTS")
    print("=" * 70)

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'Pair F0.5':>12} "
        f"{'Macro F0.5':>12} "
        f"{'Predicted':>12}"
    )

    print("-" * 70)

    results = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            threshold,
            validation_s1_ids,
            probability_map,
            ground_truth,
        )

        results.append(result)

        print(
            f"{threshold:>10.3f} "
            f"{result['precision']:>12.4f} "
            f"{result['recall']:>10.4f} "
            f"{result['pair_f05']:>12.4f} "
            f"{result['macro_f05']:>12.4f} "
            f"{result['predicted_positive_pairs']:>12,}"
        )

    # --------------------------------------------------------
    # Best threshold
    # --------------------------------------------------------

    best_result = max(
        results,
        key=lambda row: row["macro_f05"],
    )

    print()
    print("=" * 70)
    print("BEST HARD-NEGATIVE MODEL THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold             : "
        f"{best_result['threshold']:.3f}"
    )

    print(
        f"Precision             : "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall                : "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"Pair F0.5             : "
        f"{best_result['pair_f05']:.4f}"
    )

    print(
        f"Macro F0.5            : "
        f"{best_result['macro_f05']:.4f}"
    )

    print(
        f"TP                    : "
        f"{best_result['tp']:,}"
    )

    print(
        f"FP                    : "
        f"{best_result['fp']:,}"
    )

    print(
        f"FN                    : "
        f"{best_result['fn']:,}"
    )

    print(
        f"Predicted positives   : "
        f"{best_result['predicted_positive_pairs']:,}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULT_PATH,
        index=False,
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        RESULT_PATH
    )

    print()
    print("=" * 70)
    print("HARD-NEGATIVE THRESHOLD SWEEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()