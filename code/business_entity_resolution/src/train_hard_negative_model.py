from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_PATH = (
    PROJECT_ROOT
    / "output"
    / "hard_negative_pair_features.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "hard_negative_logistic_model.joblib"
)

METRIC_PATH = (
    PROJECT_ROOT
    / "output"
    / "hard_negative_model_info.json"
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRAINING HARD-NEGATIVE LOGISTIC MODEL")
    print("=" * 70)

    print(
        f"Feature file : {FEATURE_PATH}"
    )

    print(
        f"Model output : {MODEL_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading feature dataset...")

    df = pd.read_parquet(
        FEATURE_PATH
    )

    print(
        f"Total rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = {
        "s1_id",
        "candidate_id",
        "label",
        "split",
    } | set(FEATURE_COLUMNS)

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing columns:\n"
            + "\n".join(sorted(missing))
        )

    # --------------------------------------------------------
    # Train / validation
    # --------------------------------------------------------

    train_df = df[
        df["split"] == "train"
    ].copy()

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    print()
    print("Dataset split:")
    print(
        f"  Train rows      : {len(train_df):,}"
    )
    print(
        f"  Validation rows : {len(validation_df):,}"
    )

    print()
    print("Training labels:")
    print(
        f"  Positive : "
        f"{int(train_df['label'].sum()):,}"
    )
    print(
        f"  Negative : "
        f"{int((train_df['label'] == 0).sum()):,}"
    )

    print()
    print("Validation labels:")
    print(
        f"  Positive : "
        f"{int(validation_df['label'].sum()):,}"
    )
    print(
        f"  Negative : "
        f"{int((validation_df['label'] == 0).sum()):,}"
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    train_s1 = set(
        train_df["s1_id"].astype(str)
    )

    validation_s1 = set(
        validation_df["s1_id"].astype(str)
    )

    overlap = (
        train_s1
        & validation_s1
    )

    if overlap:
        raise RuntimeError(
            "S1 leakage detected between training "
            "and validation sets."
        )

    print()
    print(
        "Grouped S1 leakage check: PASS"
    )

    # --------------------------------------------------------
    # Prepare X / y
    # --------------------------------------------------------

    X_train = (
        train_df[FEATURE_COLUMNS]
        .astype(float)
    )

    y_train = (
        train_df["label"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print(
        "Training Logistic Regression..."
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

    print(
        "Model training complete."
    )

    # --------------------------------------------------------
    # Training diagnostics
    # --------------------------------------------------------

    train_probabilities = (
        model.predict_proba(
            X_train
        )[:, 1]
    )

    positive_probabilities = (
        train_probabilities[y_train.to_numpy() == 1]
    )

    negative_probabilities = (
        train_probabilities[y_train.to_numpy() == 0]
    )

    print()
    print(
        "Training probability diagnostics:"
    )

    print(
        f"  Positive mean probability: "
        f"{positive_probabilities.mean():.6f}"
    )

    print(
        f"  Negative mean probability: "
        f"{negative_probabilities.mean():.6f}"
    )

    print(
        f"  Negative max probability : "
        f"{negative_probabilities.max():.6f}"
    )

    # --------------------------------------------------------
    # Feature coefficients
    # --------------------------------------------------------

    print()
    print(
        "Model coefficients:"
    )

    coefficients = (
        model.coef_[0]
    )

    for feature, coefficient in sorted(
        zip(
            FEATURE_COLUMNS,
            coefficients,
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    ):

        print(
            f"  {feature:28s}: "
            f"{coefficient:+.6f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save model information
    # --------------------------------------------------------

    model_info = {
        "model_type": "LogisticRegression",
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "training_positive_rows": int(
            y_train.sum()
        ),
        "training_negative_rows": int(
            (y_train == 0).sum()
        ),
        "validation_positive_rows": int(
            validation_df["label"].sum()
        ),
        "validation_negative_rows": int(
            (validation_df["label"] == 0).sum()
        ),
        "feature_columns": FEATURE_COLUMNS,
        "hard_negative_rows": 844,
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
    }

    with open(
        METRIC_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            model_info,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HARD-NEGATIVE MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model  : {MODEL_PATH}"
    )

    print(
        f"Info   : {METRIC_PATH}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()