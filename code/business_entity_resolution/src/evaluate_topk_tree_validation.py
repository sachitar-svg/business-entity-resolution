from pathlib import Path
import sqlite3
import json

import joblib
import pandas as pd
import pyarrow.parquet as pq

from generate_candidate_pairs import (
    build_block_terms,
    clean_country,
    safe_text,
    quote_fts_term,
)
from build_pair_features import (
    calculate_features,
    prepare_record,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE1_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "train_source1_normalized.parquet"
)

SOURCE2_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "train_source2_normalized.parquet"
)

SOURCE3_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "train_source3_normalized.parquet"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "train_ground_truth.parquet"
)

VALIDATION_IDS_PATH = (
    PROJECT_ROOT
    / "output"
    / "tree_validation_s1_ids.csv"
)

INDEX_PATH = (
    PROJECT_ROOT
    / "output"
    / "blocking_v2_index.sqlite"
)

NAME_STATS_PATH = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "data"
    / "token_stats"
    / "name_token_counts.json"
)

ADDRESS_STATS_PATH = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "data"
    / "token_stats"
    / "address_token_counts.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "tree_matching_model.joblib"
)

TOP_K_VALUES = [
    100,
    200,
    300,
]

THRESHOLD = 0.951

BATCH_SIZE = 5000


# ============================================================
# F0.5
# ============================================================

def calculate_f05(
    precision,
    recall,
):
    beta_squared = 0.25

    if precision == 0.0 and recall == 0.0:
        return 0.0

    denominator = (
        beta_squared * precision
        + recall
    )

    if denominator == 0.0:
        return 0.0

    return (
        (1.0 + beta_squared)
        * precision
        * recall
        / denominator
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(path):

    gt = pd.read_parquet(path)

    result = {}

    for _, row in gt.iterrows():

        s1_id = safe_text(
            row["source1_entity_id"]
        )

        raw = row["matched_entity_ids"]

        if raw is None or pd.isna(raw):

            matches = set()

        else:

            text = str(raw).strip()

            if text:

                matches = {
                    value.strip()
                    for value in text.split(",")
                    if value.strip()
                }

            else:

                matches = set()

        result[s1_id] = matches

    return result


# ============================================================
# VALIDATION S1
# ============================================================

def load_validation_ids():

    df = pd.read_csv(
        VALIDATION_IDS_PATH
    )

    if "s1_id" not in df.columns:

        raise ValueError(
            "Validation ID file must contain "
            "'s1_id'."
        )

    return (
        df["s1_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )


def load_validation_rows():

    target_ids = set(
        load_validation_ids()
    )

    parquet_file = pq.ParquetFile(
        SOURCE1_PATH
    )

    rows = {}

    for batch in parquet_file.iter_batches(
        batch_size=100_000,
        columns=[
            "entity_id",
            "country",
            "business_name",
            "business_address",
            "norm_name",
            "norm_address",
            "compact_name",
        ],
    ):

        for row in batch.to_pylist():

            s1_id = safe_text(
                row["entity_id"]
            )

            if s1_id in target_ids:

                rows[s1_id] = row

                if len(rows) == len(target_ids):
                    return rows

    return rows


# ============================================================
# TOP-K QUERY
# ============================================================

def fetch_ranked_candidates(
    connection,
    terms,
    country,
    max_k,
):
    """
    Return candidates ordered by FTS5 BM25.

    Lower BM25 score = stronger FTS match.
    """

    if not terms:
        return []

    query = " OR ".join(
        quote_fts_term(term)
        for term in sorted(terms)
    )

    rows = connection.execute(
        """
        SELECT
            c.entity_id,
            bm25(block_fts) AS score
        FROM block_fts AS f
        INNER JOIN candidate_records AS c
            ON c.rowid = f.rowid
        WHERE block_fts MATCH ?
          AND c.country = ?
        ORDER BY score ASC
        LIMIT ?
        """,
        (
            query,
            country,
            max_k,
        ),
    ).fetchall()

    return [
        row[0]
        for row in rows
    ]


# ============================================================
# LOAD REQUIRED CANDIDATE RECORDS
# ============================================================

def collect_candidate_ids(
    ranked_candidates,
):

    result = set()

    for candidates in ranked_candidates.values():

        result.update(
            candidates
        )

    return result


def load_candidate_records(
    required_ids,
):

    records = {}

    for source_path in (
        SOURCE2_PATH,
        SOURCE3_PATH,
    ):

        parquet_file = pq.ParquetFile(
            source_path
        )

        for batch in parquet_file.iter_batches(
            batch_size=100_000,
            columns=[
                "entity_id",
                "country",
                "business_name",
                "business_address",
                "norm_name",
                "norm_address",
                "compact_name",
            ],
        ):

            for row in batch.to_pylist():

                entity_id = safe_text(
                    row["entity_id"]
                )

                if entity_id in required_ids:

                    records[entity_id] = row

            if len(records) >= len(required_ids):

                return records

    return records


# ============================================================
# SCORE TOP-K
# ============================================================

def evaluate_top_k(
    top_k,
    ranked_candidates,
    s1_rows,
    candidate_records,
    ground_truth,
    model,
    feature_columns,
):
    """
    Score only the TOP-K candidates per S1 entity.
    """

    total_candidate_links = 0
    total_gt_pairs = 0
    total_captured_gt = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    macro_f05_values = []

    for s1_id, candidate_list in ranked_candidates.items():

        # ----------------------------------------------------
        # Restrict to K
        # ----------------------------------------------------

        candidates = candidate_list[:top_k]

        total_candidate_links += len(
            candidates
        )

        s1_row = s1_rows[s1_id]

        gt = ground_truth.get(
            s1_id,
            set(),
        )

        total_gt_pairs += len(gt)

        captured = (
            gt
            &
            set(candidates)
        )

        total_captured_gt += len(
            captured
        )

        # ----------------------------------------------------
        # Build feature rows in batches
        # ----------------------------------------------------

        predicted_matches = set()

        feature_batch = []

        def score_batch():

            nonlocal predicted_matches

            if not feature_batch:
                return

            feature_df = pd.DataFrame(
                feature_batch
            )

            X = feature_df[
                feature_columns
            ]

            probabilities = (
                model.predict_proba(X)[:, 1]
            )

            for candidate_id, probability in zip(
                feature_df["candidate_id"],
                probabilities,
            ):

                if probability >= THRESHOLD:

                    predicted_matches.add(
                        candidate_id
                    )

            feature_batch.clear()

        for candidate_id in candidates:

            candidate_row = candidate_records.get(
                candidate_id
            )

            if candidate_row is None:
                continue

            feature_batch.append(
    calculate_features(
        prepare_record(s1_row),
        prepare_record(candidate_row),
        0,
        "validation",
    )
)

            if len(feature_batch) >= BATCH_SIZE:
                score_batch()

        score_batch()

        # ----------------------------------------------------
        # Per-S1 metrics
        # ----------------------------------------------------

        tp = len(
            predicted_matches & gt
        )

        fp = len(
            predicted_matches - gt
        )

        fn = len(
            gt - predicted_matches
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        macro_f05_values.append(
            calculate_f05(
                precision,
                recall,
            )
        )

    # --------------------------------------------------------
    # Global metrics
    # --------------------------------------------------------

    precision = (
        total_tp
        / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp
        / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    pair_f05 = calculate_f05(
        precision,
        recall,
    )

    macro_f05 = (
        sum(macro_f05_values)
        / len(macro_f05_values)
        if macro_f05_values
        else 0.0
    )

    candidate_recall = (
        total_captured_gt
        / total_gt_pairs
        if total_gt_pairs > 0
        else 0.0
    )

    return {
        "top_k": top_k,
        "mean_candidates": (
            total_candidate_links
            / len(ranked_candidates)
        ),
        "candidate_links": total_candidate_links,
        "candidate_recall": candidate_recall,
        "precision": precision,
        "recall": recall,
        "pair_f05": pair_f05,
        "macro_f05": macro_f05,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TOP-K + TREE V3 END-TO-END VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load statistics
    # --------------------------------------------------------

    with open(
        NAME_STATS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        name_counts = json.load(file)

    with open(
        ADDRESS_STATS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        address_counts = json.load(file)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_bundle = joblib.load(
        MODEL_PATH
    )

    model = model_bundle["model"]

    feature_columns = model_bundle[
        "feature_columns"
    ]

    print(
        f"\nModel features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    s1_rows = load_validation_rows()

    ground_truth = load_ground_truth(
        GROUND_TRUTH_PATH
    )

    print(
        f"Validation S1 entities: "
        f"{len(s1_rows):,}"
    )

    # --------------------------------------------------------
    # Build ranked TOP-3000 candidates once
    # --------------------------------------------------------

    connection = sqlite3.connect(
        INDEX_PATH
    )

    try:

        ranked_candidates = {}

        for index, (
            s1_id,
            s1_row,
        ) in enumerate(
            s1_rows.items(),
            start=1,
        ):

            terms = build_block_terms(
                safe_text(
                    s1_row["norm_name"]
                ),
                safe_text(
                    s1_row["compact_name"]
                ),
                safe_text(
                    s1_row["norm_address"]
                ),
                name_counts,
                address_counts,
            )

            ranked_candidates[s1_id] = (
                fetch_ranked_candidates(
                    connection,
                    terms,
                    clean_country(
                        s1_row["country"]
                    ),
                    max(TOP_K_VALUES),
                )
            )

            if index % 25 == 0:

                print(
                    f"Ranked S1: "
                    f"{index:,}"
                )

    finally:

        connection.close()

    # --------------------------------------------------------
    # Load only candidate records actually needed
    # --------------------------------------------------------

    required_candidate_ids = (
        collect_candidate_ids(
            ranked_candidates
        )
    )

    print(
        "\nUnique candidate records needed: "
        f"{len(required_candidate_ids):,}"
    )

    candidate_records = (
        load_candidate_records(
            required_candidate_ids
        )
    )

    print(
        "Candidate records loaded: "
        f"{len(candidate_records):,}"
    )

    # --------------------------------------------------------
    # Evaluate each TOP-K
    # --------------------------------------------------------

    results = []

    for top_k in TOP_K_VALUES:

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"EVALUATING TOP-K = {top_k:,}"
        )

        result = evaluate_top_k(
            top_k,
            ranked_candidates,
            s1_rows,
            candidate_records,
            ground_truth,
            model,
            feature_columns,
        )

        results.append(
            result
        )

        print(
            f"Mean candidates/S1 : "
            f"{result['mean_candidates']:,.2f}"
        )

        print(
            f"Candidate recall    : "
            f"{result['candidate_recall']:.4%}"
        )

        print(
            f"Precision           : "
            f"{result['precision']:.4%}"
        )

        print(
            f"Recall              : "
            f"{result['recall']:.4%}"
        )

        print(
            f"Pair F0.5           : "
            f"{result['pair_f05']:.4f}"
        )

        print(
            f"Macro F0.5          : "
            f"{result['macro_f05']:.4f}"
        )

        print(
            f"TP: {result['tp']}"
            f" | FP: {result['fp']}"
            f" | FN: {result['fn']}"
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL TOP-K + TREE V3 COMPARISON"
    )

    print(
        "=" * 70
    )

    result_df = pd.DataFrame(
        results
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        "\nCurrent full-candidate Tree V3 reference:"
    )

    print(
        "Precision = 0.9268"
    )

    print(
        "Recall    = 0.7192"
    )

    print(
        "Pair F0.5 = 0.8762"
    )

    print(
        "Macro F0.5 = 0.8167"
    )


if __name__ == "__main__":
    main()