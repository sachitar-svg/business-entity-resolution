from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FP_PATH = (
    PROJECT_ROOT
    / "output"
    / "v2_false_positives.csv"
)

MODEL_FN_PATH = (
    PROJECT_ROOT
    / "output"
    / "v2_model_false_negatives.csv"
)

BLOCKER_MISS_PATH = (
    PROJECT_ROOT
    / "output"
    / "v2_blocker_misses.csv"
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
]


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize_feature_file(
    path,
    label,
):
    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    if not path.exists():
        print(
            f"File not found: {path}"
        )
        return

    df = pd.read_csv(path)

    print(
        f"Rows: {len(df):,}"
    )

    if df.empty:
        print("No rows.")
        return

    if "model_probability" in df.columns:
        print()
        print("Probability:")
        print(
            f"  Mean : "
            f"{df['model_probability'].mean():.6f}"
        )
        print(
            f"  Min  : "
            f"{df['model_probability'].min():.6f}"
        )
        print(
            f"  Max  : "
            f"{df['model_probability'].max():.6f}"
        )

    print()
    print("Feature means:")

    available_features = [
        column
        for column in FEATURE_COLUMNS
        if column in df.columns
    ]

    for column in available_features:

        print(
            f"  {column:28s}: "
            f"{df[column].mean():.6f}"
        )

    print()
    print("Feature occurrence counts:")

    for column in [
        "name_exact",
        "name_compact_exact",
        "number_overlap",
        "country_match",
    ]:

        if column not in df.columns:
            continue

        count = int(
            df[column].sum()
        )

        print(
            f"  {column:28s}: "
            f"{count:,} / {len(df):,}"
        )

    print()
    print("Similarity buckets:")

    for column in [
        "name_char_similarity",
        "address_char_similarity",
        "name_token_jaccard",
        "address_token_jaccard",
        "number_jaccard",
    ]:

        if column not in df.columns:
            continue

        values = df[column]

        print(
            f"\n  {column}"
        )

        print(
            f"    >= 0.90 : "
            f"{int((values >= 0.90).sum()):,}"
        )

        print(
            f"    >= 0.75 : "
            f"{int((values >= 0.75).sum()):,}"
        )

        print(
            f"    >= 0.50 : "
            f"{int((values >= 0.50).sum()):,}"
        )

        print(
            f"    >= 0.25 : "
            f"{int((values >= 0.25).sum()):,}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V2 ERROR PATTERN SUMMARY")
    print("=" * 70)

    print(
        "\nThese summaries use only the already-generated "
        "error-analysis CSV files."
    )

    summarize_feature_file(
        FP_PATH,
        "FALSE POSITIVES",
    )

    summarize_feature_file(
        MODEL_FN_PATH,
        "MODEL FALSE NEGATIVES",
    )

    # --------------------------------------------------------
    # Blocker misses
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BLOCKER MISSES")
    print("=" * 70)

    if BLOCKER_MISS_PATH.exists():

        blocker_df = pd.read_csv(
            BLOCKER_MISS_PATH
        )

        print(
            f"Rows: {len(blocker_df):,}"
        )

        if not blocker_df.empty:

            print(
                f"Unique S1 entities: "
                f"{blocker_df['s1_id'].nunique():,}"
            )

            print()
            print(
                blocker_df
                .head(20)
                .to_string(index=False)
            )

    else:

        print(
            f"File not found: "
            f"{BLOCKER_MISS_PATH}"
        )

    print()
    print("=" * 70)
    print("V2 ERROR SUMMARY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()