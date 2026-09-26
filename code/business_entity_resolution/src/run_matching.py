import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


# Make the src directory importable when this script is run directly.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Build_matching_features import calculate_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run final business entity matching."
    )

    parser.add_argument(
        "--candidate-file",
        required=True,
        help="Candidate pairs TSV file."
    )

    parser.add_argument(
        "--s1-file",
        required=True,
        help="Normalized Source 1 Parquet file."
    )

    parser.add_argument(
        "--s2-file",
        required=True,
        help="Normalized Source 2 Parquet file."
    )

    parser.add_argument(
        "--s3-file",
        required=True,
        help="Normalized Source 3 Parquet file."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Trained logistic regression model bundle."
    )

    parser.add_argument(
        "--threshold",
        required=True,
        type=float,
        help="Probability threshold for accepting matches."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output TSV file."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("FINAL BUSINESS ENTITY MATCHING")
    print("=" * 70)

    print(f"\nCandidate file : {args.candidate_file}")
    print(f"S1 file        : {args.s1_file}")
    print(f"S2 file        : {args.s2_file}")
    print(f"S3 file        : {args.s3_file}")
    print(f"Model          : {args.model}")
    print(f"Threshold      : {args.threshold}")
    print(f"Output         : {args.output}")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading model...")

    model_bundle = joblib.load(args.model)

    if not isinstance(model_bundle, dict):
        raise ValueError(
            "Expected the model file to contain a dictionary bundle."
        )

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    print("Model loaded.")
    print(f"Expected features: {len(feature_columns)}")

    # --------------------------------------------------------
    # LOAD NORMALIZED DATA
    # --------------------------------------------------------

    print("\nLoading Source 1...")

    s1 = pd.read_parquet(args.s1_file)

    print(f"Source 1 records: {len(s1):,}")

    print("\nLoading Source 2...")

    s2 = pd.read_parquet(args.s2_file)

    print(f"Source 2 records: {len(s2):,}")

    print("\nLoading Source 3...")

    s3 = pd.read_parquet(args.s3_file)

    print(f"Source 3 records: {len(s3):,}")

    # --------------------------------------------------------
    # LOAD CANDIDATES
    # --------------------------------------------------------

    print("\nLoading candidate file...")

    candidates = pd.read_csv(
        args.candidate_file,
        sep="\t"
    )

    print(f"Source 1 candidate rows: {len(candidates):,}")

    print("\nInitial loading complete.")

    # --------------------------------------------------------
    # EXPAND CANDIDATE LISTS
    # --------------------------------------------------------

    print("\nExpanding candidate lists...")

    candidates = candidates.rename(
        columns={
            "source1_entity_id": "s1_id",
            "candidate_entity_ids": "candidate_ids",
        }
    )

    candidates["candidate_ids"] = candidates[
        "candidate_ids"
    ].fillna("")

    candidates["candidate_id"] = candidates[
        "candidate_ids"
    ].str.split(",")

    candidate_pairs = candidates[
        ["s1_id", "candidate_id"]
    ].explode(
        "candidate_id"
    )

    candidate_pairs["candidate_id"] = (
        candidate_pairs["candidate_id"]
        .astype(str)
        .str.strip()
    )

    candidate_pairs = candidate_pairs[
        candidate_pairs["candidate_id"] != ""
    ].reset_index(drop=True)

    print(
        f"Candidate pairs after expansion: "
        f"{len(candidate_pairs):,}"
    )

    print("\nFirst 5 candidate pairs:")

    print(
        candidate_pairs
        .head()
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # BUILD CANDIDATE LOOKUP
    # --------------------------------------------------------

    print("\nBuilding candidate lookup...")

    s1_lookup = s1.set_index("entity_id", drop=False)
    s2_lookup = s2.set_index("entity_id", drop=False)
    s3_lookup = s3.set_index("entity_id", drop=False)

    print("Candidate lookup ready.")

     # --------------------------------------------------------
    # CALCULATE FEATURES
    # --------------------------------------------------------

    print("\nCalculating matching features...")

    feature_rows = []

    for _, pair in candidate_pairs.iterrows():
        s1_id = pair["s1_id"]
        candidate_id = pair["candidate_id"]

        if s1_id not in s1_lookup.index:
            continue

        if candidate_id in s2_lookup.index:
            candidate_row = s2_lookup.loc[candidate_id]
        elif candidate_id in s3_lookup.index:
            candidate_row = s3_lookup.loc[candidate_id]
        else:
            continue

        s1_row = s1_lookup.loc[s1_id]

        features = calculate_features(
            s1_row,
            candidate_row
        )

        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)

    print(
        f"Feature rows created: "
        f"{len(feature_df):,}"
    )

    print("\nFeature columns:")
    print(feature_df.columns.tolist())

    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    print("\nPreparing model features...")

    X = feature_df[feature_columns]

    print("Running logistic regression...")

    feature_df["probability"] = model.predict_proba(X)[:, 1]

    print("Predictions complete.")

    print("\nProbability summary:")
    print(feature_df["probability"].describe())

    # --------------------------------------------------------
    # APPLY MATCH THRESHOLD
    # --------------------------------------------------------

    print(f"\nApplying threshold: {args.threshold}")

    matches = feature_df[
        feature_df["probability"] >= args.threshold
    ].copy()

    print(
        f"Matched pairs: "
        f"{len(matches):,}"
    )

    print(
        f"Unique S1 entities with matches: "
        f"{matches['s1_id'].nunique():,}"
    )

    print("\nFirst 10 matches:")

    print(
        matches[
            ["s1_id", "candidate_id", "probability"]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    print("\nSaving matching results...")

    output_columns = [
        "s1_id",
        "candidate_id",
        "probability",
    ]

    matches[output_columns].to_csv(
        args.output,
        sep="\t",
        index=False
    )

    print(f"Results saved to: {args.output}")

    print("\n" + "=" * 70)
    print("MATCHING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()