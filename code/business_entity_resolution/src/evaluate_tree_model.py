from pathlib import Path
import argparse
import csv
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = PROJECT_ROOT / "code" / "business_entity_resolution" / "src"

FEATURE_PATH = (
    PROJECT_ROOT
    / "output"
    / "hard_negative_pair_features.parquet"
)

TREE_MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "tree_matching_model.joblib"
)

DEFAULT_S1_IDS_PATH = (
    PROJECT_ROOT
    / "output"
    / "tree_validation_s1_ids.csv"
)

DEFAULT_CANDIDATE_PATH = (
    PROJECT_ROOT
    / "output"
    / "tree_validation_candidate_pairs.tsv"
)

DEFAULT_RESULTS_PATH = (
    PROJECT_ROOT
    / "output"
    / "tree_validation_threshold_sweep.csv"
)

CANDIDATE_GENERATOR = (
    SRC_DIR
    / "generate_candidate_pairs.py"
)

GROUND_TRUTH_REQUIRED = {
    "source1_entity_id",
    "matched_entity_ids",
}


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


BATCH_SIZE = 10_000


# ============================================================
# IMPORT FEATURE HELPERS
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_pair_features import (
    find_ground_truth_file,
    load_ground_truth,
    load_selected_candidate_records,
    load_selected_s1_records,
    calculate_features,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_text(value):
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def load_validation_s1_ids(feature_path):
    """
    Recover the exact 200-S1 validation split used by the
    hard-negative model training.
    """

    df = pd.read_parquet(
        feature_path,
        columns=["s1_id", "split"],
    )

    validation_ids = sorted(
        {
            safe_text(value)
            for value in df.loc[
                df["split"] == "validation",
                "s1_id",
            ].tolist()
            if safe_text(value)
        }
    )

    if len(validation_ids) != 200:
        raise RuntimeError(
            "Expected exactly 200 validation S1 entities, "
            f"but found {len(validation_ids):,}."
        )

    return validation_ids


def save_s1_ids(ids, path):
    """
    Write validation S1 IDs in the CSV format expected by
    generate_candidate_pairs.py: one column named s1_id.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["s1_id"])

        for s1_id in ids:
            writer.writerow([s1_id])


def ensure_candidate_file(
    validation_ids,
    ids_path,
    candidate_path,
):
    """
    Generate the exact V2 blocker candidates for the 200-S1
    validation set using the already-built disk-backed index.

    Existing matching candidate file is reused when possible.
    """

    if candidate_path.exists():
        print(
            f"\nCandidate file already exists:\n"
            f"{candidate_path}"
        )
        return

    save_s1_ids(
        validation_ids,
        ids_path,
    )

    if not CANDIDATE_GENERATOR.exists():
        raise FileNotFoundError(
            "Candidate generator not found:\n"
            f"{CANDIDATE_GENERATOR}"
        )

    command = [
        sys.executable,
        str(CANDIDATE_GENERATOR),
        "--s1-ids-file",
        str(ids_path),
        "--output",
        str(candidate_path),
    ]

    print("\nGenerating validation candidate pairs...")
    print("The existing blocking index will be reused.")
    print(
        "Command:",
        " ".join(command),
        flush=True,
    )

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    if not candidate_path.exists():
        raise RuntimeError(
            "Candidate generation completed but output file "
            "was not created."
        )


def read_candidate_file(candidate_path):
    """
    Read grouped candidate TSV:

        source1_entity_id<TAB>candidate_entity_ids

    Returns:
        candidate_rows:
            list of (s1_id, candidate_id)
        candidate_ids:
            unique candidate IDs
        candidate_links:
            total candidate links
    """

    candidate_rows = []
    candidate_ids = set()
    candidate_s1_ids = []
    total_links = 0

    with open(
        candidate_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        header = file.readline().rstrip("\r\n").split("\t")

        if header != [
            "source1_entity_id",
            "candidate_entity_ids",
        ]:
            raise ValueError(
                "Unexpected candidate-file header:\n"
                + "\t".join(header)
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
            candidate_text = safe_text(parts[1])

            if not s1_id:
                continue

            candidate_s1_ids.append(s1_id)

            if not candidate_text:
                continue

            ids = [
                value.strip()
                for value in candidate_text.split(",")
                if value.strip()
            ]

            # Remove duplicates while preserving row content.
            ids = list(dict.fromkeys(ids))

            total_links += len(ids)

            for candidate_id in ids:
                candidate_rows.append(
                    (
                        s1_id,
                        candidate_id,
                    )
                )

                candidate_ids.add(candidate_id)

    return (
        candidate_rows,
        candidate_s1_ids,
        candidate_ids,
        total_links,
    )


def fbeta_from_counts(
    tp,
    fp,
    fn,
    beta=0.5,
):
    denominator = (
        (1.0 + beta * beta) * tp
        + (beta * beta) * fn
        + fp
    )

    if denominator == 0:
        return 1.0

    return (
        (1.0 + beta * beta) * tp
        / denominator
    )


def evaluate_threshold(
    probabilities,
    labels,
    s1_indices,
    entity_count,
    threshold,
):
    predicted = (
        probabilities >= threshold
    )

    true_positive = (
        predicted
        & (labels == 1)
    )

    false_positive = (
        predicted
        & (labels == 0)
    )

    false_negative = (
        (~predicted)
        & (labels == 1)
    )

    tp = int(true_positive.sum())
    fp = int(false_positive.sum())
    fn = int(false_negative.sum())

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 1.0
    )

    pair_f05 = fbeta_from_counts(
        tp,
        fp,
        fn,
        beta=0.5,
    )

    tp_by = np.bincount(
        s1_indices,
        weights=true_positive.astype(np.float64),
        minlength=entity_count,
    )

    fp_by = np.bincount(
        s1_indices,
        weights=false_positive.astype(np.float64),
        minlength=entity_count,
    )

    fn_by = np.bincount(
        s1_indices,
        weights=false_negative.astype(np.float64),
        minlength=entity_count,
    )

    macro_scores = []

    for entity_index in range(entity_count):
        macro_scores.append(
            fbeta_from_counts(
                tp_by[entity_index],
                fp_by[entity_index],
                fn_by[entity_index],
                beta=0.5,
            )
        )

    macro_f05 = float(
        np.mean(macro_scores)
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "pair_f05": float(pair_f05),
        "macro_f05": float(macro_f05),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_matches": int(predicted.sum()),
    }


# ============================================================
# TREE ERROR ANALYSIS
# ============================================================

def _collect_detailed_examples(
    selected_indices,
    probabilities,
    labels,
    candidate_rows,
    s1_indices,
    s1_records,
    candidate_records,
    error_type,
    threshold,
):
    """Build detailed rows for selected TP/FP/FN examples."""

    rows = []

    for index in selected_indices:
        index = int(index)
        s1_id, candidate_id = candidate_rows[index]

        s1_record = s1_records[s1_id]
        candidate_record = candidate_records[candidate_id]
        label = int(labels[index])

        features = calculate_features(
            s1_record,
            candidate_record,
            label,
            "validation",
        )

        row = {
            "s1_id": s1_id,
            "candidate_id": candidate_id,
            "probability": float(probabilities[index]),
            "threshold": float(threshold),
            "true_label": label,
            "predicted_label": int(probabilities[index] >= threshold),
            "error_type": error_type,
            "margin_from_threshold": float(
                abs(probabilities[index] - threshold)
            ),
        }

        for column in FEATURE_COLUMNS:
            row[column] = float(features[column])

        rows.append(row)

    return rows


def write_tree_error_analysis(
    probabilities,
    labels,
    s1_indices,
    validation_s1_ids,
    candidate_rows,
    s1_records,
    candidate_records,
    threshold,
    output_dir,
):
    """Write actionable TP/FP/FN analysis for the selected threshold."""

    predicted = probabilities >= threshold

    tp_indices = np.flatnonzero(
        predicted & (labels == 1)
    )
    fp_indices = np.flatnonzero(
        predicted & (labels == 0)
    )
    fn_indices = np.flatnonzero(
        (~predicted) & (labels == 1)
    )

    selected_rows = []
    selected_rows.extend(
        _collect_detailed_examples(
            tp_indices,
            probabilities,
            labels,
            candidate_rows,
            s1_indices,
            s1_records,
            candidate_records,
            "TP",
            threshold,
        )
    )
    selected_rows.extend(
        _collect_detailed_examples(
            fp_indices,
            probabilities,
            labels,
            candidate_rows,
            s1_indices,
            s1_records,
            candidate_records,
            "FP",
            threshold,
        )
    )
    selected_rows.extend(
        _collect_detailed_examples(
            fn_indices,
            probabilities,
            labels,
            candidate_rows,
            s1_indices,
            s1_records,
            candidate_records,
            "FN",
            threshold,
        )
    )

    detailed_df = pd.DataFrame(selected_rows)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_path = output_dir / "tree_validation_tp_fp_fn_details.csv"
    detailed_df.to_csv(
        detailed_path,
        index=False,
    )

    # Feature patterns across TP / FP / FN.
    feature_summary_rows = []

    for error_type in ("TP", "FP", "FN"):
        subset = detailed_df.loc[
            detailed_df["error_type"] == error_type
        ]

        if subset.empty:
            continue

        row = {
            "group": error_type,
            "count": int(len(subset)),
            "mean_probability": float(
                subset["probability"].mean()
            ),
            "median_probability": float(
                subset["probability"].median()
            ),
        }

        for column in FEATURE_COLUMNS:
            row[f"mean_{column}"] = float(
                subset[column].mean()
            )

        feature_summary_rows.append(row)

    feature_summary_path = (
        output_dir / "tree_validation_error_feature_summary.csv"
    )
    pd.DataFrame(feature_summary_rows).to_csv(
        feature_summary_path,
        index=False,
    )

    # Per-S1 difficulty summary. This is especially important because
    # the final metric is macro F0.5 across Source-1 entities.
    per_s1 = []

    for entity_index, s1_id in enumerate(validation_s1_ids):
        mask = s1_indices == entity_index
        entity_labels = labels[mask]
        entity_predicted = predicted[mask]

        tp = int((entity_predicted & (entity_labels == 1)).sum())
        fp = int((entity_predicted & (entity_labels == 0)).sum())
        fn = int((~entity_predicted & (entity_labels == 1)).sum())

        denominator = 0.25 * (tp + fp) + fn
        f05 = (1.25 * tp / denominator) if denominator > 0 else 1.0

        per_s1.append({
            "s1_id": s1_id,
            "candidate_links": int(mask.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f05": float(f05),
        })

    per_s1_df = pd.DataFrame(per_s1).sort_values(
        ["f05", "fn", "fp"],
        ascending=[True, False, False],
    )

    per_s1_path = output_dir / "tree_validation_per_s1_error_summary.csv"
    per_s1_df.to_csv(
        per_s1_path,
        index=False,
    )

    # Hardest false negatives and false positives near/above the threshold.
    fn_path = output_dir / "tree_validation_top_false_negatives.csv"
    fp_path = output_dir / "tree_validation_top_false_positives.csv"

    fn_df = detailed_df.loc[
        detailed_df["error_type"] == "FN"
    ].sort_values(
        ["probability", "margin_from_threshold"],
        ascending=[False, True],
    )

    fp_df = detailed_df.loc[
        detailed_df["error_type"] == "FP"
    ].sort_values(
        ["probability", "margin_from_threshold"],
        ascending=[False, True],
    )

    fn_df.head(50).to_csv(fn_path, index=False)
    fp_df.head(50).to_csv(fp_path, index=False)

    print(
        "\nTREE MODEL ERROR ANALYSIS"
    )
    print(
        "-" * 70
    )
    print(
        f"Analysis threshold : {threshold:.3f}"
    )
    print(
        f"TP: {len(tp_indices):,} | "
        f"FP: {len(fp_indices):,} | "
        f"FN: {len(fn_indices):,}"
    )
    print(
        f"Detailed errors    : {detailed_path}"
    )
    print(
        f"Feature summary    : {feature_summary_path}"
    )
    print(
        f"Per-S1 summary     : {per_s1_path}"
    )
    print(
        f"Top FNs            : {fn_path}"
    )
    print(
        f"Top FPs            : {fp_path}"
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_tree_model(
    feature_path,
    model_path,
    candidate_path,
    output_path,
):
    print("=" * 70)
    print("TREE MODEL FULL-CANDIDATE VALIDATION")
    print("=" * 70)

    print(
        f"\nFeature dataset : {feature_path}"
    )
    print(
        f"Tree model      : {model_path}"
    )
    print(
        f"Candidate file  : {candidate_path}"
    )

    # --------------------------------------------------------
    # Validation entities
    # --------------------------------------------------------

    print(
        "\nLoading exact 200-S1 validation split..."
    )

    validation_s1_ids = load_validation_s1_ids(
        feature_path
    )

    validation_s1_set = set(
        validation_s1_ids
    )

    s1_to_index = {
        s1_id: index
        for index, s1_id in enumerate(
            validation_s1_ids
        )
    }

    print(
        f"Validation S1 entities: "
        f"{len(validation_s1_ids):,}"
    )

    # --------------------------------------------------------
    # Candidate pairs
    # --------------------------------------------------------

    print(
        "\nReading validation candidate file..."
    )

    (
        candidate_rows,
        candidate_s1_ids,
        candidate_ids,
        total_candidate_links,
    ) = read_candidate_file(
        candidate_path
    )

    candidate_s1_set = set(
        candidate_s1_ids
    )

    if candidate_s1_set != validation_s1_set:
        missing = validation_s1_set - candidate_s1_set
        extra = candidate_s1_set - validation_s1_set

        raise RuntimeError(
            "Candidate file does not match the exact "
            "200-S1 validation split.\n"
            f"Missing S1 IDs : {len(missing):,}\n"
            f"Extra S1 IDs   : {len(extra):,}"
        )

    print(
        f"Candidate links     : "
        f"{total_candidate_links:,}"
    )

    print(
        f"Unique candidate IDs: "
        f"{len(candidate_ids):,}"
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth_path = find_ground_truth_file(
        PROJECT_ROOT
    )

    print(
        f"\nGround truth: {ground_truth_path}"
    )

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading tree model..."
    )

    artifact = joblib.load(
        model_path
    )

    # The training script stores a dictionary artifact containing
    # the fitted estimator plus metadata. Unwrap the estimator so
    # prediction uses the actual classifier object.
    if hasattr(artifact, "predict_proba"):
        model = artifact
    elif isinstance(artifact, dict):
        model = None

        # Prefer the conventional estimator keys.
        for key in (
            "model",
            "classifier",
            "estimator",
            "tree_model",
            "matching_model",
        ):
            candidate = artifact.get(key)

            if hasattr(candidate, "predict_proba"):
                model = candidate
                print(
                    f"Model artifact unwrapped from key: {key}"
                )
                break

        # Fallback: search all dictionary values for the estimator.
        if model is None:
            for key, candidate in artifact.items():
                if hasattr(candidate, "predict_proba"):
                    model = candidate
                    print(
                        f"Model artifact unwrapped from key: {key}"
                    )
                    break

        if model is None:
            raise TypeError(
                "Loaded tree model artifact is a dictionary, but "
                "no value with predict_proba() was found. "
                f"Available keys: {list(artifact.keys())}"
            )
    else:
        raise TypeError(
            "Loaded tree model is not a classifier object or "
            f"dictionary artifact. Type: {type(artifact).__name__}"
        )

    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    print(
        "\nLoading Source-1 validation records..."
    )

    s1_records = load_selected_s1_records(
        validation_s1_set
    )

    if len(s1_records) != len(validation_s1_set):
        raise RuntimeError(
            "Not all validation Source-1 records "
            "were found in normalized Parquet."
        )

    print(
        f"Loaded S1 records: "
        f"{len(s1_records):,}"
    )

    print(
        "\nLoading all candidate records needed "
        "for this validation set..."
    )

    candidate_records = load_selected_candidate_records(
        candidate_ids
    )

    print(
        f"Loaded candidate records: "
        f"{len(candidate_records):,}"
    )

    missing_candidate_ids = (
        candidate_ids
        - set(candidate_records)
    )

    if missing_candidate_ids:
        raise RuntimeError(
            "Candidate records missing from normalized Parquet: "
            f"{len(missing_candidate_ids):,}"
        )

    # --------------------------------------------------------
    # Score all candidate links
    # --------------------------------------------------------

    print(
        "\nScoring full candidate set..."
    )
    print(
        f"Scoring links: {total_candidate_links:,}",
        flush=True,
    )

    probability_parts = []
    label_parts = []
    s1_index_parts = []

    batch_features = []
    batch_labels = []
    batch_s1_indices = []

    scored_links = 0

    def flush_batch():
        nonlocal batch_features
        nonlocal batch_labels
        nonlocal batch_s1_indices
        nonlocal scored_links

        if not batch_features:
            return

        # The tree was trained with a pandas DataFrame, so keep the
        # feature names here to avoid sklearn's feature-name warning.
        X = pd.DataFrame(
            batch_features,
            columns=FEATURE_COLUMNS,
            dtype=np.float32,
        )

        probabilities = model.predict_proba(
            X
        )[:, 1]

        probability_parts.append(
            probabilities.astype(
                np.float32
            )
        )

        label_parts.append(
            np.asarray(
                batch_labels,
                dtype=np.uint8,
            )
        )

        s1_index_parts.append(
            np.asarray(
                batch_s1_indices,
                dtype=np.int16,
            )
        )

        scored_links += len(
            batch_features
        )

        if (
            scored_links % 100_000 < BATCH_SIZE
            or scored_links == total_candidate_links
        ):
            print(
                f"  Scored {scored_links:,} / "
                f"{total_candidate_links:,}",
                flush=True,
            )

        batch_features = []
        batch_labels = []
        batch_s1_indices = []

    for s1_id, candidate_id in candidate_rows:

        s1_record = s1_records.get(
            s1_id
        )

        candidate_record = candidate_records.get(
            candidate_id
        )

        if (
            s1_record is None
            or candidate_record is None
        ):
            raise RuntimeError(
                "Record lookup failed for pair:\n"
                f"S1={s1_id}\n"
                f"Candidate={candidate_id}"
            )

        true_ids = ground_truth.get(
            s1_id,
            set(),
        )

        label = int(
            candidate_id in true_ids
        )

        features = calculate_features(
            s1_record,
            candidate_record,
            label,
            "validation",
        )

        batch_features.append(
            [
                float(features[column])
                for column in FEATURE_COLUMNS
            ]
        )

        batch_labels.append(
            label
        )

        batch_s1_indices.append(
            s1_to_index[s1_id]
        )

        if len(batch_features) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    probabilities = np.concatenate(
        probability_parts
    )

    labels = np.concatenate(
        label_parts
    )

    s1_indices = np.concatenate(
        s1_index_parts
    )

    if len(probabilities) != total_candidate_links:
        raise RuntimeError(
            "Scored-link count does not match "
            "candidate-link count.\n"
            f"Expected: {total_candidate_links:,}\n"
            f"Scored:   {len(probabilities):,}"
        )

    # --------------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------------

    # Fine sweep around the current coarse optimum (0.95).
    # Use 0.001 resolution from 0.940 through 0.970 inclusive.
    thresholds = [
        round(0.940 + 0.001 * index, 3)
        for index in range(31)
    ]

    results = []

    print(
        "\n" + "=" * 70
    )
    print(
        "TREE MODEL THRESHOLD SWEEP"
    )
    print(
        "=" * 70
    )

    for threshold in thresholds:

        result = evaluate_threshold(
            probabilities,
            labels,
            s1_indices,
            len(validation_s1_ids),
            threshold,
        )

        results.append(
            result
        )

        print(
            f"{threshold:>6.3f} | "
            f"P {result['precision']:.4f} | "
            f"R {result['recall']:.4f} | "
            f"Pair F0.5 {result['pair_f05']:.4f} | "
            f"Macro F0.5 {result['macro_f05']:.4f} | "
            f"TP {result['tp']:,} | "
            f"FP {result['fp']:,} | "
            f"FN {result['fn']:,}"
        )

    results_df = pd.DataFrame(
        results
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Best macro-F0.5 threshold
    # --------------------------------------------------------

    best = results_df.loc[
        results_df["macro_f05"].idxmax()
    ]

    print(
        "\n" + "=" * 70
    )
    print(
        "BEST TREE MODEL THRESHOLD"
    )
    print(
        "=" * 70
    )

    print(
        f"Threshold       : {best['threshold']:.3f}"
    )
    print(
        f"Precision       : {best['precision']:.4f}"
    )
    print(
        f"Recall          : {best['recall']:.4f}"
    )
    print(
        f"Pair F0.5       : {best['pair_f05']:.4f}"
    )
    print(
        f"Macro F0.5      : {best['macro_f05']:.4f}"
    )
    print(
        f"TP              : {int(best['tp']):,}"
    )
    print(
        f"FP              : {int(best['fp']):,}"
    )
    print(
        f"FN              : {int(best['fn']):,}"
    )
    print(
        f"Predicted matches: "
        f"{int(best['predicted_matches']):,}"
    )

    print(
        f"\nThreshold results saved to:\n"
        f"{output_path}"
    )

    # --------------------------------------------------------
    # Error analysis at the selected threshold
    # --------------------------------------------------------

    write_tree_error_analysis(
        probabilities,
        labels,
        s1_indices,
        validation_s1_ids,
        candidate_rows,
        s1_records,
        candidate_records,
        float(best["threshold"]),
        output_path.parent,
    )

    print(
        "\nDone."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the tree-based matching model on the "
            "same 200-S1 full candidate validation set used "
            "for the Logistic V2 comparison."
        )
    )

    parser.add_argument(
        "--feature-file",
        default=str(FEATURE_PATH),
    )

    parser.add_argument(
        "--model",
        default=str(TREE_MODEL_PATH),
    )

    parser.add_argument(
        "--s1-ids-file",
        default=str(DEFAULT_S1_IDS_PATH),
    )

    parser.add_argument(
        "--candidate-file",
        default=str(DEFAULT_CANDIDATE_PATH),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULTS_PATH),
    )

    args = parser.parse_args()

    feature_path = Path(
        args.feature_file
    )

    model_path = Path(
        args.model
    )

    ids_path = Path(
        args.s1_ids_file
    )

    candidate_path = Path(
        args.candidate_file
    )

    output_path = Path(
        args.output
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{feature_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Tree model not found:\n{model_path}"
        )

    validation_s1_ids = load_validation_s1_ids(
        feature_path
    )

    save_s1_ids(
        validation_s1_ids,
        ids_path,
    )

    ensure_candidate_file(
        validation_s1_ids,
        ids_path,
        candidate_path,
    )

    evaluate_tree_model(
        feature_path,
        model_path,
        candidate_path,
        output_path,
    )


if __name__ == "__main__":
    main()
