from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier


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
    / "tree_matching_model.joblib"
)

INFO_PATH = (
    PROJECT_ROOT
    / "output"
    / "tree_matching_model_info.json"
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
    print("TRAINING TREE-BASED MATCHING MODEL")
    print("=" * 70)

    print(
        f"Feature file : {FEATURE_PATH}"
    )

    print(
        f"Model output : {MODEL_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "Loading feature dataset..."
    )

    df = pd.read_parquet(
        FEATURE_PATH
    )

    print(
        f"Total rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = {
        "s1_id",
        "candidate_id",
        "label",
        "split",
    } | set(FEATURE_COLUMNS)

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                sorted(missing_columns)
            )
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_df = df[
        df["split"] == "train"
    ].copy()

    validation_df = df[
        df["split"] == "validation"
    ].copy()

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
            "S1 leakage detected."
        )

    print()
    print("Dataset split:")

    print(
        f"  Train rows      : "
        f"{len(train_df):,}"
    )

    print(
        f"  Validation rows : "
        f"{len(validation_df):,}"
    )

    print(
        f"  Train S1        : "
        f"{len(train_s1):,}"
    )

    print(
        f"  Validation S1   : "
        f"{len(validation_s1):,}"
    )

    print(
        "  S1 leakage      : PASS"
    )

    print()
    print("Labels:")

    print(
        f"  Train positive : "
        f"{int(train_df['label'].sum()):,}"
    )

    print(
        f"  Train negative : "
        f"{int((train_df['label'] == 0).sum()):,}"
    )

    print(
        f"  Valid positive : "
        f"{int(validation_df['label'].sum()):,}"
    )

    print(
        f"  Valid negative : "
        f"{int((validation_df['label'] == 0).sum()):,}"
    )

    # --------------------------------------------------------
    # Prepare training data
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
    # Train model
    # --------------------------------------------------------

    print()
    print(
        "Training HistGradientBoostingClassifier..."
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=False,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Tree model training complete."
    )

    # --------------------------------------------------------
    # Training diagnostics
    # --------------------------------------------------------

    train_probabilities = (
        model.predict_proba(
            X_train
        )[:, 1]
    )

    train_positive = (
        train_probabilities[
            y_train.to_numpy() == 1
        ]
    )

    train_negative = (
        train_probabilities[
            y_train.to_numpy() == 0
        ]
    )

    print()
    print(
        "Training probability diagnostics:"
    )

    print(
        f"  Positive mean : "
        f"{train_positive.mean():.6f}"
    )

    print(
        f"  Negative mean : "
        f"{train_negative.mean():.6f}"
    )

    print(
        f"  Negative max  : "
        f"{train_negative.max():.6f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )

    model_info = {
        "model_type": (
            "HistGradientBoostingClassifier"
        ),
        "training_rows": int(
            len(train_df)
        ),
        "validation_rows": int(
            len(validation_df)
        ),
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
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "random_state": 42,
    }

    with open(
        INFO_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            model_info,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TREE MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model : {MODEL_PATH}"
    )

    print(
        f"Info  : {INFO_PATH}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()