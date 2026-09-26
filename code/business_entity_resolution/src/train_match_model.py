from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, fbeta_score


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_PATH = (
    PROJECT_ROOT
    / "output"
    / "benchmark_pair_features.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "baseline_logistic_model.joblib"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "output"
    / "benchmark_validation_scores.parquet"
)

METRIC_PATH = (
    PROJECT_ROOT
    / "output"
    / "baseline_validation_metrics.json"
)


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
# ENTITY-LEVEL MACRO F0.5
# ============================================================

def calculate_macro_f05(
    dataframe: pd.DataFrame,
    threshold: float,
) -> float:
    """
    Calculate macro F0.5 across Source-1 entities.

    Each S1 entity gets its own F0.5 score.
    The final score is the mean across S1 entities.
    """

    scores = []

    for s1_id, group in dataframe.groupby("s1_id", sort=False):

        y_true = group["label"].to_numpy()

        y_pred = (
            group["probability"].to_numpy() >= threshold
        ).astype(int)

        score = fbeta_score(
            y_true,
            y_pred,
            beta=0.5,
            zero_division=0,
        )

        scores.append(score)

    if not scores:
        return 0.0

    return float(np.mean(scores))


# ============================================================
# PAIR-LEVEL METRICS
# ============================================================

def calculate_pair_metrics(
    y_true,
    y_pred,
):
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f05 = fbeta_score(
        y_true,
        y_pred,
        beta=0.5,
        zero_division=0,
    )

    return precision, recall, f05


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRAINING BASELINE MATCHING MODEL")
    print("=" * 70)

    print(f"Feature file : {FEATURE_PATH}")
    print(f"Model output : {MODEL_PATH}")
    print()

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    print("Loading feature dataset...")

    df = pd.read_parquet(FEATURE_PATH)

    print(f"Total rows  : {len(df):,}")
    print(f"Total cols  : {len(df.columns):,}")
    print()

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = (
        [
            "s1_id",
            "candidate_id",
            "label",
            "split",
        ]
        + FEATURE_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "validation"].copy()

    print("Dataset split:")
    print(f"  Train rows      : {len(train_df):,}")
    print(f"  Validation rows : {len(valid_df):,}")
    print()

    print("Train labels:")
    print(
        f"  Positive : {int(train_df['label'].sum()):,}"
    )
    print(
        f"  Negative : {int((train_df['label'] == 0).sum()):,}"
    )
    print()

    print("Validation labels:")
    print(
        f"  Positive : {int(valid_df['label'].sum()):,}"
    )
    print(
        f"  Negative : {int((valid_df['label'] == 0).sum()):,}"
    )
    print()

    # --------------------------------------------------------
    # Prepare X / y
    # --------------------------------------------------------

    X_train = train_df[FEATURE_COLUMNS].astype(float)
    y_train = train_df["label"].astype(int)

    X_valid = valid_df[FEATURE_COLUMNS].astype(float)
    y_valid = valid_df["label"].astype(int)

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("Model training complete.")
    print()

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    print("Scoring validation pairs...")

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    valid_df["probability"] = probabilities

    print("Validation scoring complete.")
    print()

    # --------------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------------

    print("=" * 70)
    print("THRESHOLD EVALUATION")
    print("=" * 70)

    threshold_results = []

    thresholds = np.arange(
        0.10,
        0.96,
        0.05,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision, recall, pair_f05 = (
            calculate_pair_metrics(
                y_valid,
                predictions,
            )
        )

        macro_f05 = calculate_macro_f05(
            valid_df,
            threshold,
        )

        predicted_positive_count = int(
            predictions.sum()
        )

        result = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "pair_f05": float(pair_f05),
            "macro_f05": float(macro_f05),
            "predicted_positive_pairs": (
                predicted_positive_count
            ),
        }

        threshold_results.append(result)

        print(
            f"Threshold {threshold:.2f} | "
            f"Precision {precision:.4f} | "
            f"Recall {recall:.4f} | "
            f"Pair F0.5 {pair_f05:.4f} | "
            f"Macro F0.5 {macro_f05:.4f} | "
            f"Predicted {predicted_positive_count:,}"
        )

    # --------------------------------------------------------
    # Select threshold using validation macro F0.5
    # --------------------------------------------------------

    best_result = max(
        threshold_results,
        key=lambda item: item["macro_f05"],
    )

    best_threshold = best_result["threshold"]

    print()
    print("=" * 70)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold             : "
        f"{best_threshold:.2f}"
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
        f"Predicted positives   : "
        f"{best_result['predicted_positive_pairs']:,}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": best_threshold,
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save validation scores
    # --------------------------------------------------------

    valid_df.to_parquet(
        RESULT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "training_positive_rows": int(y_train.sum()),
        "validation_positive_rows": int(y_valid.sum()),
        "best_threshold": float(best_threshold),
        "best_precision": float(
            best_result["precision"]
        ),
        "best_recall": float(
            best_result["recall"]
        ),
        "best_pair_f05": float(
            best_result["pair_f05"]
        ),
        "best_macro_f05": float(
            best_result["macro_f05"]
        ),
        "threshold_results": threshold_results,
        "feature_columns": FEATURE_COLUMNS,
    }

    with open(
        METRIC_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(f"Model                   : {MODEL_PATH}")
    print(f"Validation scores       : {RESULT_PATH}")
    print(f"Metrics                 : {METRIC_PATH}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()