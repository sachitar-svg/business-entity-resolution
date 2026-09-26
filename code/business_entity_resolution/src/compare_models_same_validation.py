from pathlib import Path
import csv
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


from build_pair_features import (
    calculate_features,
    load_ground_truth,
    load_selected_candidate_records,
    load_selected_s1_records,
    safe_text,
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

HARD_FEATURE_PATH = (
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

V2_MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "hard_negative_logistic_model.joblib"
)

FAIR_V1_MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "fair_baseline_same_validation.joblib"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "output"
    / "fair_model_comparison_results.csv"
)


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
# THRESHOLDS
# ============================================================

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


# Candidate rows can contain very large comma-separated fields.
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

def entity_f05(
    true_ids,
    predicted_ids,
):
    true_ids = set(true_ids)
    predicted_ids = set(predicted_ids)

    tp = len(
        true_ids & predicted_ids
    )

    fp = len(
        predicted_ids - true_ids
    )

    fn = len(
        true_ids - predicted_ids
    )

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
# TRAIN FAIR V1
# ============================================================

def train_fair_v1(
    base_df,
    validation_s1_ids,
):
    """
    Train a baseline Logistic Regression on exactly the same
    S1 training population used by Model V2.

    No hard negatives are added.
    """

    training_df = base_df[
        ~base_df["s1_id"].astype(str).isin(
            validation_s1_ids
        )
    ].copy()

    training_s1_ids = set(
        training_df["s1_id"].astype(str)
    )

    overlap = (
        training_s1_ids
        & validation_s1_ids
    )

    if overlap:
        raise RuntimeError(
            "Validation leakage detected in fair V1."
        )

    X_train = (
        training_df[FEATURE_COLUMNS]
        .astype(float)
    )

    y_train = (
        training_df["label"]
        .astype(int)
    )

    print(
        "\nFAIR V1 TRAINING DATA"
    )

    print(
        f"  Training S1 entities : "
        f"{len(training_s1_ids):,}"
    )

    print(
        f"  Training rows        : "
        f"{len(training_df):,}"
    )

    print(
        f"  Positive rows        : "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"  Negative rows        : "
        f"{int((y_train == 0).sum()):,}"
    )

    print(
        "  S1 leakage check     : PASS"
    )

    print(
        "\nTraining fair baseline Logistic Regression..."
    )

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        FAIR_V1_MODEL_PATH,
    )

    print(
        f"Fair V1 saved to: "
        f"{FAIR_V1_MODEL_PATH}"
    )

    return model


# ============================================================
# SCORE ALL CANDIDATES
# ============================================================

def score_all_candidates(
    validation_s1_ids,
    candidate_map,
    s1_records,
    candidate_records,
    v1_model,
    v2_model,
):
    """
    Calculate pair features once and obtain probabilities from
    both models.
    """

    probabilities = {}

    total_links = sum(
        len(ids)
        for ids in candidate_map.values()
    )

    print()
    print(
        f"Candidate links to score: "
        f"{total_links:,}"
    )

    for index, s1_id in enumerate(
        sorted(validation_s1_ids),
        start=1,
    ):

        s1_record = s1_records.get(
            s1_id
        )

        if s1_record is None:
            probabilities[s1_id] = []
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
                    "fair_comparison",
                )
            )

            usable_candidate_ids.append(
                candidate_id
            )

        if not feature_rows:

            probabilities[s1_id] = []

        else:

            feature_df = pd.DataFrame(
                feature_rows
            )

            X = (
                feature_df[
                    FEATURE_COLUMNS
                ]
                .astype(float)
            )

            v1_probabilities = (
                v1_model.predict_proba(X)[:, 1]
            )

            v2_probabilities = (
                v2_model.predict_proba(X)[:, 1]
            )

            probabilities[s1_id] = [
                (
                    candidate_id,
                    float(v1_probability),
                    float(v2_probability),
                )
                for candidate_id,
                v1_probability,
                v2_probability
                in zip(
                    usable_candidate_ids,
                    v1_probabilities,
                    v2_probabilities,
                )
            ]

        if index % 25 == 0:
            print(
                f"Scored S1: "
                f"{index:,}/{len(validation_s1_ids):,}",
                flush=True,
            )

    return probabilities


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_name,
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

        if model_name == "fair_v1":

            predicted_ids = {
                candidate_id
                for candidate_id,
                v1_probability,
                v2_probability
                in scored_candidates
                if v1_probability >= threshold
            }

        elif model_name == "v2_hard_negative":

            predicted_ids = {
                candidate_id
                for candidate_id,
                v1_probability,
                v2_probability
                in scored_candidates
                if v2_probability >= threshold
            }

        else:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

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

    denominator = (
        1.25 * total_tp
        + 0.25 * total_fn
        + total_fp
    )

    pair_f05 = (
        1.25 * total_tp
        / denominator
        if denominator
        else 1.0
    )

    return {
        "model": model_name,
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
    print("FAIR V1 VS HARD-NEGATIVE V2 COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
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

    print(
        "\nLoading hard-negative feature dataset..."
    )

    hard_df = pd.read_parquet(
        HARD_FEATURE_PATH,
        columns=[
            "s1_id",
            "split",
        ],
    )

    # --------------------------------------------------------
    # New validation group B
    # --------------------------------------------------------

    validation_s1_ids = set(
        hard_df.loc[
            hard_df["split"] == "validation",
            "s1_id",
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"\nValidation S1 entities: "
        f"{len(validation_s1_ids):,}"
    )

    # --------------------------------------------------------
    # Fair V1
    # --------------------------------------------------------

    fair_v1_model = train_fair_v1(
        base_df,
        validation_s1_ids,
    )

    # --------------------------------------------------------
    # V2 model
    # --------------------------------------------------------

    print(
        "\nLoading hard-negative V2 model..."
    )

    v2_bundle = joblib.load(
        V2_MODEL_PATH
    )

    v2_model = v2_bundle["model"]

    v2_feature_columns = (
        v2_bundle["feature_columns"]
    )

    if v2_feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "V2 feature columns do not match expected columns."
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

    # --------------------------------------------------------
    # Candidates
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
    # Score once
    # --------------------------------------------------------

    print(
        "\nScoring candidates with BOTH models..."
    )

    probability_map = score_all_candidates(
        validation_s1_ids,
        candidate_map,
        s1_records,
        candidate_records,
        fair_v1_model,
        v2_model,
    )

    print(
        "\nAll candidate probabilities generated."
    )

    # --------------------------------------------------------
    # Evaluate both models
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAME-VALIDATION MODEL COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<20}"
        f"{'Threshold':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'Pair F0.5':>12}"
        f"{'Macro F0.5':>12}"
        f"{'Predicted':>12}"
    )

    print("-" * 90)

    results = []

    for model_name in [
        "fair_v1",
        "v2_hard_negative",
    ]:

        for threshold in THRESHOLDS:

            result = evaluate_model(
                model_name,
                threshold,
                validation_s1_ids,
                probability_map,
                ground_truth,
            )

            results.append(result)

            print(
                f"{model_name:<20}"
                f"{threshold:>10.3f}"
                f"{result['precision']:>12.4f}"
                f"{result['recall']:>10.4f}"
                f"{result['pair_f05']:>12.4f}"
                f"{result['macro_f05']:>12.4f}"
                f"{result['predicted_positive_pairs']:>12,}"
            )

    # --------------------------------------------------------
    # Best result for each model
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("BEST RESULT — FAIR V1")
    print("=" * 70)

    best_v1 = (
        result_df[
            result_df["model"] == "fair_v1"
        ]
        .sort_values(
            "macro_f05",
            ascending=False,
        )
        .iloc[0]
    )

    print(
        f"Threshold  : "
        f"{best_v1['threshold']:.3f}"
    )

    print(
        f"Precision  : "
        f"{best_v1['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{best_v1['recall']:.4f}"
    )

    print(
        f"Pair F0.5  : "
        f"{best_v1['pair_f05']:.4f}"
    )

    print(
        f"Macro F0.5 : "
        f"{best_v1['macro_f05']:.4f}"
    )

    print(
        f"TP         : "
        f"{int(best_v1['tp']):,}"
    )

    print(
        f"FP         : "
        f"{int(best_v1['fp']):,}"
    )

    print(
        f"FN         : "
        f"{int(best_v1['fn']):,}"
    )

    print()
    print("=" * 70)
    print("BEST RESULT — HARD-NEGATIVE V2")
    print("=" * 70)

    best_v2 = (
        result_df[
            result_df["model"] == "v2_hard_negative"
        ]
        .sort_values(
            "macro_f05",
            ascending=False,
        )
        .iloc[0]
    )

    print(
        f"Threshold  : "
        f"{best_v2['threshold']:.3f}"
    )

    print(
        f"Precision  : "
        f"{best_v2['precision']:.4f}"
    )

    print(
        f"Recall     : "
        f"{best_v2['recall']:.4f}"
    )

    print(
        f"Pair F0.5  : "
        f"{best_v2['pair_f05']:.4f}"
    )

    print(
        f"Macro F0.5 : "
        f"{best_v2['macro_f05']:.4f}"
    )

    print(
        f"TP         : "
        f"{int(best_v2['tp']):,}"
    )

    print(
        f"FP         : "
        f"{int(best_v2['fp']):,}"
    )

    print(
        f"FN         : "
        f"{int(best_v2['fn']):,}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_df.to_csv(
        RESULT_PATH,
        index=False,
    )

    print()
    print(
        f"Comparison saved to:"
    )

    print(
        RESULT_PATH
    )

    print()
    print("=" * 70)
    print("FAIR MODEL COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()