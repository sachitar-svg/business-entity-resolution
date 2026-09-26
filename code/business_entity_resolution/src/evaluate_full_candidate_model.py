from pathlib import Path
import csv
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
)

from build_pair_features import (
    load_ground_truth,
    load_selected_s1_records,
    load_selected_candidate_records,
    calculate_features,
    safe_text,
)
# Allow very large candidate_entity_ids fields in the TSV.
csv.field_size_limit(sys.maxsize)

# ============================================================
# PATHS
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
    / "benchmark_full_candidate_predictions.tsv"
)

THRESHOLD = 0.45


# ============================================================
# READ CANDIDATES
# ============================================================

def load_validation_candidates(
    candidate_path,
    validation_s1_ids,
):
    """
    Read the complete candidate list, but keep only
    validation Source-1 entities.
    """

    candidate_map = {}

    with open(
        candidate_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        required = {
            "source1_entity_id",
            "candidate_entity_ids",
        }

        if not reader.fieldnames:
            raise ValueError(
                "Candidate file has no header."
            )

        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "Candidate file is missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            s1_id = safe_text(
                row.get("source1_entity_id")
            )

            if s1_id not in validation_s1_ids:
                continue

            candidate_text = safe_text(
                row.get("candidate_entity_ids")
            )

            candidate_ids = [
                value.strip()
                for value in candidate_text.split(",")
                if value.strip()
            ]

            candidate_map[s1_id] = candidate_ids

    return candidate_map


# ============================================================
# ENTITY-LEVEL F0.5
# ============================================================

def entity_f05(
    true_ids,
    predicted_ids,
):
    """
    Calculate F0.5 for one Source-1 entity.

    The universe includes:
      - all generated candidates
      - any ground-truth IDs missed by blocking

    This ensures blocker misses count as false negatives.
    """

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

    if tp == 0:
        return 0.0, tp, fp, fn

    score = (
        1.25 * tp
        / (
            1.25 * tp
            + 0.25 * fn
            + fp
        )
    )

    return score, tp, fp, fn


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FULL CANDIDATE MODEL EVALUATION")
    print("=" * 70)

    print(f"Feature file : {FEATURE_PATH}")
    print(f"Candidate    : {CANDIDATE_PATH}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Threshold    : {THRESHOLD:.2f}")
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
        f"Feature columns loaded: "
        f"{len(feature_columns)}"
    )

    print()

    # --------------------------------------------------------
    # Load validation S1 IDs
    # --------------------------------------------------------

    print(
        "Loading validation Source-1 IDs..."
    )

    feature_df = pd.read_parquet(
        FEATURE_PATH,
        columns=[
            "s1_id",
            "split",
        ],
    )

    validation_s1_ids = set(
        feature_df.loc[
            feature_df["split"] == "validation",
            "s1_id",
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"Validation S1 entities: "
        f"{len(validation_s1_ids):,}"
    )

    del feature_df

    # --------------------------------------------------------
    # Load ground truth
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
    # Load complete candidate lists
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
        f"Validation candidate S1 rows : "
        f"{len(candidate_map):,}"
    )

    print(
        f"Validation candidate links    : "
        f"{total_candidate_links:,}"
    )

    print(
        f"Unique candidate IDs          : "
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

    candidate_records = load_selected_candidate_records(
        unique_candidate_ids
    )

    print(
        f"Candidate records loaded: "
        f"{len(candidate_records):,}"
    )

    # --------------------------------------------------------
    # Score every candidate
    # --------------------------------------------------------

    print(
        "\nScoring ALL validation candidate pairs..."
    )

    entity_results = []

    all_true = []
    all_pred = []

    macro_scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

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
                    "benchmark",
                )
            )

            usable_candidate_ids.append(
                candidate_id
            )

        if feature_rows:

            feature_matrix = pd.DataFrame(
                feature_rows
            )[feature_columns].astype(float)

            probabilities = (
                model.predict_proba(
                    feature_matrix
                )[:, 1]
            )

            predicted_ids = {
                candidate_id
                for candidate_id, probability
                in zip(
                    usable_candidate_ids,
                    probabilities,
                )
                if probability >= THRESHOLD
            }

        else:
            predicted_ids = set()

        true_ids = set(
            ground_truth.get(
                s1_id,
                set(),
            )
        )

        score, tp, fp, fn = entity_f05(
            true_ids,
            predicted_ids,
        )

        macro_scores.append(score)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        entity_results.append(
            {
                "source1_entity_id": s1_id,
                "true_match_count": len(true_ids),
                "candidate_count": len(candidate_ids),
                "predicted_match_count": len(
                    predicted_ids
                ),
                "true_matches_captured": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "f05": score,
                "predicted_entity_ids": ",".join(
                    sorted(predicted_ids)
                ),
            }
        )

        if index % 25 == 0:
            print(
                f"Processed validation S1: "
                f"{index:,}/{len(validation_s1_ids):,}",
                flush=True,
            )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    macro_f05 = float(
        np.mean(macro_scores)
        if macro_scores
        else 0.0
    )

    precision = (
        total_tp
        / (total_tp + total_fp)
        if (total_tp + total_fp)
        else 0.0
    )

    recall = (
        total_tp
        / (total_tp + total_fn)
        if (total_tp + total_fn)
        else 0.0
    )

    pair_f05 = (
        fbeta_score(
            [1] * total_tp
            + [0] * total_fp
            + [1] * total_fn,
            [1] * total_tp
            + [1] * total_fp
            + [0] * total_fn,
            beta=0.5,
            zero_division=0,
        )
        if (
            total_tp
            + total_fp
            + total_fn
        )
        else 0.0
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        entity_results
    )

    result_df.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FULL CANDIDATE EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Validation S1 entities : "
        f"{len(entity_results):,}"
    )

    print(
        f"Candidate links scored : "
        f"{total_candidate_links:,}"
    )

    print(
        f"TP                     : "
        f"{total_tp:,}"
    )

    print(
        f"FP                     : "
        f"{total_fp:,}"
    )

    print(
        f"FN                     : "
        f"{total_fn:,}"
    )

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

    print(
        f"Threshold              : "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"\nPredictions saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("\nDone.")


if __name__ == "__main__":
    main()