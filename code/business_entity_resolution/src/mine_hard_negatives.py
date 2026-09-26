from pathlib import Path
import csv
import heapq
import sys

import joblib
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
    / "benchmark_pair_features.parquet"
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
    / "baseline_logistic_model.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "mined_hard_negatives.parquet"
)

# Mine the top difficult negatives for every S1.
HARD_NEGATIVES_PER_S1 = 20

# Ignore low-confidence negatives.
MIN_PROBABILITY = 0.80

# Large candidate fields are expected.
csv.field_size_limit(sys.maxsize)


# ============================================================
# LOAD COMPLETE CANDIDATES FOR SELECTED S1
# ============================================================

def load_candidates(
    candidate_path,
    selected_s1_ids,
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
                "Candidate file must have exactly two columns."
            )

        if header[0] != "source1_entity_id":
            raise ValueError(
                "Expected column: source1_entity_id"
            )

        if header[1] != "candidate_entity_ids":
            raise ValueError(
                "Expected column: candidate_entity_ids"
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

            if s1_id not in selected_s1_ids:
                continue

            candidate_text = safe_text(
                parts[1]
            )

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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HARD-NEGATIVE MINING")
    print("=" * 70)

    print(
        f"Model                  : {MODEL_PATH}"
    )

    print(
        f"Candidate file         : {CANDIDATE_PATH}"
    )

    print(
        f"Hard negatives / S1    : "
        f"{HARD_NEGATIVES_PER_S1}"
    )

    print(
        f"Minimum probability    : "
        f"{MIN_PROBABILITY:.2f}"
    )

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading trained model...")

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
    # Identify CURRENT validation S1 entities
    # --------------------------------------------------------

    print(
        "\nLoading current validation S1 IDs..."
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
        f"Current validation S1 entities: "
        f"{len(validation_s1_ids):,}"
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    print(
        "\nLoading ground truth..."
    )

    ground_truth = load_ground_truth(
        GROUND_TRUTH_PATH
    )

    # --------------------------------------------------------
    # Load all candidates for these S1 entities
    # --------------------------------------------------------

    print(
        "\nLoading complete candidate lists..."
    )

    candidate_map = load_candidates(
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
    # Mine top hard negatives
    # --------------------------------------------------------

    print(
        "\nScoring candidates and mining hard negatives..."
    )

    hard_negative_rows = []

    total_scored = 0
    total_excluded_true_matches = 0

    for index, s1_id in enumerate(
        sorted(validation_s1_ids),
        start=1,
    ):

        s1_record = s1_records.get(
            s1_id
        )

        if s1_record is None:
            continue

        candidate_ids = candidate_map.get(
            s1_id,
            [],
        )

        if not candidate_ids:
            continue

        feature_rows = []
        usable_candidate_ids = []

        true_ids = ground_truth.get(
            s1_id,
            set(),
        )

        for candidate_id in candidate_ids:

            # Never mine a real ground-truth match
            # as a negative example.
            if candidate_id in true_ids:
                total_excluded_true_matches += 1
                continue

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
                    "hard_negative",
                )
            )

            usable_candidate_ids.append(
                candidate_id
            )

        if feature_rows:

            feature_df = pd.DataFrame(
                feature_rows
            )

            X = feature_df[
                feature_columns
            ].astype(float)

            probabilities = (
                model.predict_proba(
                    X
                )[:, 1]
            )

            # Min-heap:
            # keep only the top K probabilities.
            heap = []

            for candidate_id, probability in zip(
                usable_candidate_ids,
                probabilities,
            ):

                total_scored += 1

                probability = float(
                    probability
                )

                if probability < MIN_PROBABILITY:
                    continue

                item = (
                    probability,
                    candidate_id,
                )

                if len(heap) < HARD_NEGATIVES_PER_S1:

                    heapq.heappush(
                        heap,
                        item,
                    )

                elif item[0] > heap[0][0]:

                    heapq.heapreplace(
                        heap,
                        item,
                    )

            # Highest probability first.
            hard_negatives = sorted(
                heap,
                reverse=True,
            )

            for rank, (
                probability,
                candidate_id,
            ) in enumerate(
                hard_negatives,
                start=1,
            ):

                hard_negative_rows.append(
                    {
                        "s1_id": s1_id,
                        "candidate_id": candidate_id,
                        "label": 0,
                        "split": "hard_negative",
                        "model_probability": probability,
                        "hard_negative_rank": rank,
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
    # Save
    # --------------------------------------------------------

    hard_negative_df = pd.DataFrame(
        hard_negative_rows
    )

    hard_negative_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HARD-NEGATIVE MINING COMPLETE")
    print("=" * 70)

    print(
        f"Candidate links available   : "
        f"{total_candidate_links:,}"
    )

    print(
        f"Candidate pairs scored      : "
        f"{total_scored:,}"
    )

    print(
        f"True matches excluded       : "
        f"{total_excluded_true_matches:,}"
    )

    print(
        f"Hard negatives mined        : "
        f"{len(hard_negative_df):,}"
    )

    if not hard_negative_df.empty:

        print(
            f"S1 entities with hard negs : "
            f"{hard_negative_df['s1_id'].nunique():,}"
        )

        print(
            f"Highest probability         : "
            f"{hard_negative_df['model_probability'].max():.6f}"
        )

        print(
            f"Lowest selected probability: "
            f"{hard_negative_df['model_probability'].min():.6f}"
        )

        print(
            f"Mean probability            : "
            f"{hard_negative_df['model_probability'].mean():.6f}"
        )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()