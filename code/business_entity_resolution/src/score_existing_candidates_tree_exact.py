from pathlib import Path
import argparse
import csv
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "code" / "business_entity_resolution" / "src"

DEFAULT_CANDIDATE_PATH = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "output" / "tree_matching_model.joblib"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "matching_results_tree_exact.tsv"
TEST_S1_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "test_source1_normalized.parquet"
)

TEST_S2_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "test_source2_normalized.parquet"
)

TEST_S3_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "test_source3_normalized.parquet"
)

# IMPORTANT:
# These are the exact 13 feature columns used when Tree V3 was trained.
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

DEFAULT_THRESHOLD = 0.951
DEFAULT_CANDIDATE_CHUNK = 20_000


# ============================================================
# IMPORT THE PROJECT'S CANONICAL FEATURE BUILDER
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_pair_features import (
    calculate_features,
    prepare_record,
)

# NOTE:
# The current canonical build_pair_features.prepare_record() accepts
# ONE record argument. Pass the full pandas row directly; do not expand
# entity_id/name/address/country/compact_name into positional arguments.



# ============================================================
# HELPERS
# ============================================================

def safe_text(value):
    """Convert a nullable value to a clean string."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def load_tree_model(model_path):
    """Load the fitted Tree V3 estimator from the saved artifact."""

    artifact = joblib.load(model_path)

    if hasattr(artifact, "predict_proba"):
        return artifact

    if isinstance(artifact, dict):
        preferred_keys = (
            "model",
            "classifier",
            "estimator",
            "tree_model",
            "matching_model",
        )

        for key in preferred_keys:
            candidate = artifact.get(key)
            if hasattr(candidate, "predict_proba"):
                print(f"Model artifact unwrapped from key: {key}")
                return candidate

        for key, candidate in artifact.items():
            if hasattr(candidate, "predict_proba"):
                print(f"Model artifact unwrapped from key: {key}")
                return candidate

    raise TypeError(
        "Could not find a classifier with predict_proba() in the model artifact."
    )

TEST_S1_DF = None
TEST_CANDIDATE_DF = None


def load_test_data():
    global TEST_S1_DF
    global TEST_CANDIDATE_DF

    if TEST_S1_DF is not None and TEST_CANDIDATE_DF is not None:
        return

    print("Loading TEST Source1...")
    TEST_S1_DF = pd.read_parquet(TEST_S1_PATH)

    print("Loading TEST Source2...")
    s2 = pd.read_parquet(TEST_S2_PATH)

    print("Loading TEST Source3...")
    s3 = pd.read_parquet(TEST_S3_PATH)

    TEST_CANDIDATE_DF = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    del s2
    del s3

    for df in [TEST_S1_DF, TEST_CANDIDATE_DF]:
        for col in [
            "entity_id",
            "norm_name",
            "compact_name",
            "norm_address",
            "country",
        ]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                )

    TEST_S1_DF = TEST_S1_DF.set_index(
        "entity_id",
        drop=False,
    )

    TEST_CANDIDATE_DF = (
        TEST_CANDIDATE_DF
        .drop_duplicates(
            subset=["entity_id"],
            keep="first",
        )
        .set_index(
            "entity_id",
            drop=False,
        )
    )

    print(
        f"TEST Source1 records: {len(TEST_S1_DF):,}"
    )

    print(
        f"TEST Source2+3 records: "
        f"{len(TEST_CANDIDATE_DF):,}"
    )


def load_selected_test_s1_records(requested_ids):
    load_test_data()

    records = {}

    for entity_id in requested_ids:
        entity_id = str(entity_id)

        if entity_id not in TEST_S1_DF.index:
            continue

        row = TEST_S1_DF.loc[entity_id]

        records[entity_id] = prepare_record(row)

    return records


def load_selected_test_candidate_records(requested_ids):
    load_test_data()

    records = {}

    for entity_id in requested_ids:
        entity_id = str(entity_id)

        if entity_id not in TEST_CANDIDATE_DF.index:
            continue

        row = TEST_CANDIDATE_DF.loc[entity_id]

        records[entity_id] = prepare_record(row)

    return records
def read_candidate_chunk(file, max_links):
    """
    Read grouped candidate rows without rebuilding or modifying the blocker.

    Input format:
        source1_entity_id<TAB>candidate_entity_ids

    Returns:
        list of (s1_id, [candidate_id, ...])
    """

    rows = []
    link_count = 0

    while link_count < max_links:
        line = file.readline()

        if not line:
            break

        line = line.rstrip("\r\n")

        if not line:
            continue

        parts = line.split("\t", 1)

        if len(parts) != 2:
            raise ValueError(
                "Malformed candidate row. Expected two tab-separated fields:\n"
                + line[:500]
            )

        s1_id = safe_text(parts[0])
        candidate_field = safe_text(parts[1])

        if not s1_id:
            raise ValueError("Encountered an empty Source-1 entity ID.")

        candidate_ids = []

        if candidate_field:
            for value in candidate_field.split(","):
                candidate_id = safe_text(value)
                if candidate_id:
                    candidate_ids.append(candidate_id)

        rows.append((s1_id, candidate_ids))
        link_count += len(candidate_ids)

    return rows

# ============================================================
# TEST DATA LOADERS
# ============================================================

TEST_S1_DF = None
TEST_CANDIDATE_DF = None


def load_test_data():
    global TEST_S1_DF
    global TEST_CANDIDATE_DF

    if TEST_S1_DF is not None and TEST_CANDIDATE_DF is not None:
        return

    print("Loading TEST Source1...")
    TEST_S1_DF = pd.read_parquet(TEST_S1_PATH)

    print("Loading TEST Source2...")
    s2 = pd.read_parquet(TEST_S2_PATH)

    print("Loading TEST Source3...")
    s3 = pd.read_parquet(TEST_S3_PATH)

    TEST_CANDIDATE_DF = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    del s2
    del s3

    for df in [TEST_S1_DF, TEST_CANDIDATE_DF]:
        for col in [
            "entity_id",
            "norm_name",
            "compact_name",
            "norm_address",
            "country",
        ]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                )

    TEST_S1_DF = TEST_S1_DF.set_index(
        "entity_id",
        drop=False,
    )

    TEST_CANDIDATE_DF = (
        TEST_CANDIDATE_DF
        .drop_duplicates(
            subset=["entity_id"],
            keep="first",
        )
        .set_index(
            "entity_id",
            drop=False,
        )
    )

    print(
        f"TEST Source1 records: {len(TEST_S1_DF):,}"
    )

    print(
        f"TEST Source2+3 records: "
        f"{len(TEST_CANDIDATE_DF):,}"
    )


def load_selected_test_s1_records(requested_ids):
    load_test_data()

    records = {}

    for entity_id in requested_ids:
        entity_id = str(entity_id)

        if entity_id not in TEST_S1_DF.index:
            continue

        row = TEST_S1_DF.loc[entity_id]

        records[entity_id] = prepare_record(row)

    return records


def load_selected_test_candidate_records(requested_ids):
    load_test_data()

    records = {}

    for entity_id in requested_ids:
        entity_id = str(entity_id)

        if entity_id not in TEST_CANDIDATE_DF.index:
            continue

        row = TEST_CANDIDATE_DF.loc[entity_id]

        records[entity_id] = prepare_record(row)

    return records

def score_rows_exact(rows, model, threshold):
    """
    Score all pairs in one chunk and return predictions grouped by S1.

    This is written separately from the file reader so the candidate TSV
    remains completely unchanged.
    """

    s1_ids = [s1_id for s1_id, _ in rows]
    candidate_ids = {
        candidate_id
        for _, ids in rows
        for candidate_id in ids
    }

    s1_records = load_selected_test_s1_records(set(s1_ids))
    candidate_records = load_selected_test_candidate_records(candidate_ids)

    missing_s1 = set(s1_ids) - set(s1_records)
    if missing_s1:
        raise RuntimeError(
            "Source-1 records missing from normalized data: "
            f"{len(missing_s1):,}"
        )

    missing_candidates = candidate_ids - set(candidate_records)
    if missing_candidates:
        raise RuntimeError(
            "Candidate records missing from normalized data: "
            f"{len(missing_candidates):,}"
        )

    predicted_by_s1 = {s1_id: [] for s1_id in s1_ids}

    feature_rows = []
    feature_keys = []
    scored = 0

    def flush():
        nonlocal feature_rows
        nonlocal feature_keys
        nonlocal scored

        if not feature_rows:
            return

        X = pd.DataFrame(
            feature_rows,
            columns=FEATURE_COLUMNS,
            dtype=np.float32,
        )

        probabilities = model.predict_proba(X)[:, 1]

        for key, probability in zip(feature_keys, probabilities):
            s1_id, candidate_id = key
            probability = float(probability)

            if probability >= threshold:
                predicted_by_s1[s1_id].append(
                    (candidate_id, probability)
                )

        scored += len(feature_rows)
        feature_rows = []
        feature_keys = []

    for s1_id, candidate_id_list in rows:
        s1_record = s1_records[s1_id]

        for candidate_id in candidate_id_list:
            candidate_record = candidate_records[candidate_id]

            # Canonical feature calculation from build_pair_features.py.
            # No RapidFuzz substitution is used here.
            features = calculate_features(
                s1_record,
                candidate_record,
                0,
                "test",
            )

            feature_rows.append(
                [
                    float(features[column])
                    for column in FEATURE_COLUMNS
                ]
            )
            feature_keys.append((s1_id, candidate_id))

            if len(feature_rows) >= 10_000:
                flush()

    flush()

    return predicted_by_s1, scored


# ============================================================
# MAIN SCORING PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score an existing candidate_pairs.tsv with Tree V3 using "
            "the project's canonical SequenceMatcher-based features."
        )
    )

    parser.add_argument(
        "--candidate-file",
        default=str(DEFAULT_CANDIDATE_PATH),
        help="Existing grouped candidate_pairs.tsv file.",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Trained Tree V3 model artifact.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Match threshold (default: {DEFAULT_THRESHOLD:.3f}).",
    )

    parser.add_argument(
        "--candidate-chunk",
        type=int,
        default=DEFAULT_CANDIDATE_CHUNK,
        help=(
            "Approximate number of candidate links processed per chunk "
            f"(default: {DEFAULT_CANDIDATE_CHUNK:,})."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output matching results TSV.",
    )

    args = parser.parse_args()

    candidate_path = Path(args.candidate_file)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found:\n{candidate_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Tree model not found:\n{model_path}"
        )

    if args.threshold < 0.0 or args.threshold > 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    if args.candidate_chunk <= 0:
        raise ValueError("--candidate-chunk must be greater than zero.")

    # Candidate rows can contain very long comma-separated candidate lists.
    csv.field_size_limit(sys.maxsize)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_tree_model(model_path)

    print("=" * 70)
    print("TREE V3 EXACT FEATURE SCORING")
    print("=" * 70)
    print(f"Candidate file : {candidate_path}")
    print(f"Model          : {model_path}")
    print(f"Threshold      : {args.threshold:.3f}")
    print(f"Chunk size     : {args.candidate_chunk:,}")
    print()
    print(
        "Using the canonical build_pair_features.calculate_features() "
        "implementation."
    )
    print(
        "The existing candidate file is read only; it is NOT rebuilt."
    )
    print()

    total_s1 = 0
    total_scored = 0
    total_predicted = 0
    zero_match_s1 = 0

    with open(
        candidate_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as source_file, open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:

        header = source_file.readline().rstrip("\r\n").split("\t")

        expected_header = [
            "source1_entity_id",
            "candidate_entity_ids",
        ]

        if header != expected_header:
            raise ValueError(
                "Unexpected candidate-file header. Expected:\n"
                f"{expected_header}\n"
                f"Found:\n{header}"
            )

        writer = csv.writer(
            output_file,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writerow([
            "source1_entity_id",
            "matched_entity_ids",
        ])

        while True:
            rows = read_candidate_chunk(
                source_file,
                args.candidate_chunk,
            )

            if not rows:
                break

            predictions, scored = score_rows_exact(
                rows,
                model,
                args.threshold,
            )

            total_scored += scored

            for s1_id, _candidate_ids in rows:
                scored_matches = predictions.get(s1_id, [])

                # Preserve the original candidate order.
                matched_set = {
                    matched_candidate
                    for matched_candidate, _probability in scored_matches
                }

                matched_ids = [
                    candidate_id
                    for candidate_id in _candidate_ids
                    if candidate_id in matched_set
                ]

                # Deduplicate while preserving order.
                unique_ids = []
                seen = set()

                for candidate_id in matched_ids:
                    if candidate_id not in seen:
                        seen.add(candidate_id)
                        unique_ids.append(candidate_id)

                writer.writerow([
                    s1_id,
                    ",".join(unique_ids),
                ])

                total_s1 += 1
                total_predicted += len(unique_ids)

                if not unique_ids:
                    zero_match_s1 += 1

            if total_s1 % 1_000 < len(rows):
                print(
                    f"  Processed S1: {total_s1:,} | "
                    f"Scored links: {total_scored:,} | "
                    f"Predicted matches: {total_predicted:,}",
                    flush=True,
                )

    print()
    print("=" * 70)
    print("TREE V3 SCORING COMPLETE")
    print("=" * 70)
    print(f"S1 rows processed       : {total_s1:,}")
    print(f"Candidate links scored  : {total_scored:,}")
    print(f"Predicted match links   : {total_predicted:,}")
    print(f"S1 rows with no matches : {zero_match_s1:,}")
    print(f"Output                  : {output_path}")
    print()


if __name__ == "__main__":
    main()
