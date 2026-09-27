from pathlib import Path
import argparse
import json
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import joblib
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PROJECT SETUP
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from build_pair_features import (
    calculate_features,
    prepare_record,
)

from generate_candidate_pairs import (
    clean_country,
    safe_text,
    quote_fts_term,
)


# ============================================================
# DEFAULT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

S1_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "test_source1_normalized.parquet"
)

S2_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "test_source2_normalized.parquet"
)

S3_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "test_source3_normalized.parquet"
)

NAME_STATS_PATH = (
    PROJECT_ROOT
    / "output"
    / "test_token_stats"
    / "name_token_counts.json"
)

ADDRESS_STATS_PATH = (
    PROJECT_ROOT
    / "output"
    / "test_token_stats"
    / "address_token_counts.json"
)

INDEX_PATH = (
    PROJECT_ROOT
    / "output"
    / "blocking_v2_test_index.sqlite"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "tree_matching_model.joblib"
)

CANDIDATE_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "candidate_pairs.tsv"
)

MATCH_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "matching_results.tsv"
)


# ============================================================
# FINAL RUN CONFIGURATION
# ============================================================

TOP_K = 20
THRESHOLD = 0.951

S1_BATCH_SIZE = 500

WORKERS = 8


# ============================================================
# THREAD-LOCAL WORKER STATE
# ============================================================

WORKER_STATE = threading.local()


# ============================================================
# COMPACT BLOCKING
# ============================================================

def get_tokens(value):
    """
    Convert normalized text into unique tokens.
    """

    value = safe_text(value)

    if not value:
        return set()

    return {
        token
        for token in value.split()
        if len(token) >= 2
    }


def build_compact_terms(
    row,
    name_counts,
    address_counts,
):
    """
    Build a compact blocking query.

    Channels retained for the first-pass final run:

    1. Exact normalized name
    2. Compact normalized name
    3. Top-two rarest name tokens
    4. Top-two rarest address tokens

    Broad single-token blocking is intentionally excluded
    from this first-pass run to keep candidate volume manageable.
    """

    terms = set()

    norm_name = safe_text(
        row["norm_name"]
    )

    compact_name = safe_text(
        row["compact_name"]
    )

    norm_address = safe_text(
        row["norm_address"]
    )

    # --------------------------------------------------------
    # Exact normalized name
    # --------------------------------------------------------

    if norm_name:

        terms.add(
            "nexact_"
            + norm_name.replace(
                " ",
                "_"
            )
        )

    # --------------------------------------------------------
    # Compact name
    # --------------------------------------------------------

    if compact_name:

        terms.add(
            "ncompact_"
            + compact_name
        )

    # --------------------------------------------------------
    # Name pair
    # --------------------------------------------------------

    name_tokens = get_tokens(
        norm_name
    )

    if len(name_tokens) >= 2:

        ordered_name = sorted(
            name_tokens,
            key=lambda token: (
                name_counts.get(
                    token,
                    10**18
                ),
                token,
            ),
        )

        pair = sorted(
            ordered_name[:2]
        )

        terms.add(
            "npair_"
            + pair[0]
            + "_"
            + pair[1]
        )

    # --------------------------------------------------------
    # Address pair
    # --------------------------------------------------------

    address_tokens = get_tokens(
        norm_address
    )

    if len(address_tokens) >= 2:

        ordered_address = sorted(
            address_tokens,
            key=lambda token: (
                address_counts.get(
                    token,
                    10**18
                ),
                token,
            ),
        )

        pair = sorted(
            ordered_address[:2]
        )

        terms.add(
            "apair_"
            + pair[0]
            + "_"
            + pair[1]
        )

    return terms


# ============================================================
# WORKER INITIALIZATION
# ============================================================

def initialize_worker(
    index_path,
    name_counts,
    address_counts,
):
    """
    Initialize one independent SQLite connection
    for each worker thread.
    """

    WORKER_STATE.connection = sqlite3.connect(
        index_path
    )

    WORKER_STATE.name_counts = name_counts
    WORKER_STATE.address_counts = address_counts


# ============================================================
# CANDIDATE RETRIEVAL
# ============================================================

def fetch_top_candidates(
    row,
):
    """
    Retrieve TOP-K candidates for one S1 entity.

    Each worker uses its own SQLite connection.
    """

    connection = WORKER_STATE.connection

    name_counts = (
        WORKER_STATE.name_counts
    )

    address_counts = (
        WORKER_STATE.address_counts
    )

    terms = build_compact_terms(
        row,
        name_counts,
        address_counts,
    )

    if not terms:
        return []

    query = " OR ".join(
        quote_fts_term(term)
        for term in sorted(terms)
    )

    country = clean_country(
        row["country"]
    )

    rows = connection.execute(
        """
        SELECT
            c.entity_id
        FROM block_fts AS f
        INNER JOIN candidate_records AS c
            ON c.rowid = f.rowid
        WHERE block_fts MATCH ?
          AND c.country = ?
        ORDER BY
            f.rank ASC,
            c.entity_id ASC
        LIMIT ?
        """,
        (
            query,
            country,
            TOP_K,
        ),
    ).fetchall()

    return [
        row[0]
        for row in rows
    ]


# ============================================================
# SOURCE 1 BATCH READER
# ============================================================

def iter_s1_batches(
    path,
    batch_size,
):
    """
    Read Source 1 in batches from Parquet.
    """

    parquet_file = pq.ParquetFile(
        path
    )

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "norm_name",
            "norm_address",
            "compact_name",
        ],
    ):

        yield batch.to_pylist()


# ============================================================
# LOAD SOURCE 2 / SOURCE 3
# ============================================================

def load_candidate_data():

    print(
        "\nLoading Source 2..."
    )

    s2 = pd.read_parquet(
        S2_PATH
    )

    print(
        f"Source 2 records: "
        f"{len(s2):,}"
    )

    print(
        "\nLoading Source 3..."
    )

    s3 = pd.read_parquet(
        S3_PATH
    )

    print(
        f"Source 3 records: "
        f"{len(s3):,}"
    )

    columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "norm_name",
        "norm_address",
        "compact_name",
    ]

    s2 = s2[
        columns
    ].set_index(
        "entity_id",
        drop=False,
    )

    s3 = s3[
        columns
    ].set_index(
        "entity_id",
        drop=False,
    )

    return s2, s3


# ============================================================
# CANDIDATE RECORD LOOKUP
# ============================================================

def get_candidate_row(
    candidate_id,
    s2,
    s3,
):
    """
    Find a candidate in Source 2 or Source 3.
    """

    if candidate_id in s2.index:

        return s2.loc[
            candidate_id
        ]

    if candidate_id in s3.index:

        return s3.loc[
            candidate_id
        ]

    return None


# ============================================================
# SCORE ONE S1 ENTITY
# ============================================================

def score_s1(
    s1_row,
    candidate_ids,
    s2,
    s3,
    model,
    feature_columns,
):
    """
    Calculate the 13 model features and score candidates
    for one Source 1 entity.
    """

    feature_rows = []

    prepared_s1 = prepare_record(
        s1_row
    )

    for candidate_id in candidate_ids:

        candidate_row = get_candidate_row(
            candidate_id,
            s2,
            s3,
        )

        if candidate_row is None:
            continue

        prepared_candidate = (
            prepare_record(
                candidate_row
            )
        )

        feature_rows.append(
            calculate_features(
                prepared_s1,
                prepared_candidate,
                0,
                "test",
            )
        )

    if not feature_rows:
        return []

    feature_df = pd.DataFrame(
        feature_rows
    )

    X = feature_df[
        feature_columns
    ]

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    matches = []

    for candidate_id, probability in zip(
        feature_df["candidate_id"],
        probabilities,
    ):

        if probability >= THRESHOLD:

            matches.append(
                candidate_id
            )

    return matches


# ============================================================
# MAIN
# ============================================================

def main():

    global TOP_K
    global THRESHOLD
    global WORKERS

    parser = argparse.ArgumentParser(
        description=(
            "Final Tree V3 inference pipeline."
        )
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=(
            "Maximum candidates per S1."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=(
            "Tree V3 probability threshold."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help=(
            "Number of parallel candidate-retrieval workers."
        ),
    )

    parser.add_argument(
        "--s1-limit",
        type=int,
        default=0,
        help=(
            "Number of S1 rows to process. "
            "0 means the full test dataset."
        ),
    )

    parser.add_argument(
        "--candidate-output",
        default=str(
            CANDIDATE_OUTPUT
        ),
        help=(
            "Candidate-pairs TSV output path."
        ),
    )

    parser.add_argument(
        "--match-output",
        default=str(
            MATCH_OUTPUT
        ),
        help=(
            "Matching-results TSV output path."
        ),
    )

    args = parser.parse_args()

    TOP_K = args.top_k
    THRESHOLD = args.threshold
    WORKERS = args.workers

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if TOP_K <= 0:

        raise ValueError(
            "--top-k must be greater than 0."
        )

    if WORKERS <= 0:

        raise ValueError(
            "--workers must be greater than 0."
        )

    if not 0.0 <= THRESHOLD <= 1.0:

        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "FINAL TREE V3 INFERENCE"
    )

    print(
        "=" * 70
    )

    print(
        f"S1 file      : {S1_PATH}"
    )

    print(
        f"S2 file      : {S2_PATH}"
    )

    print(
        f"S3 file      : {S3_PATH}"
    )

    print(
        f"Candidate cap: TOP-{TOP_K}"
    )

    print(
        f"Threshold    : {THRESHOLD}"
    )

    print(
        f"Workers      : {WORKERS}"
    )

    if args.s1_limit > 0:

        print(
            f"S1 limit     : "
            f"{args.s1_limit:,}"
        )

    else:

        print(
            "S1 limit     : FULL DATASET"
        )

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    required_files = [
        S1_PATH,
        S2_PATH,
        S3_PATH,
        NAME_STATS_PATH,
        ADDRESS_STATS_PATH,
        INDEX_PATH,
        MODEL_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load token statistics
    # --------------------------------------------------------

    print(
        "\nLoading test token statistics..."
    )

    with open(
        NAME_STATS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        name_counts = json.load(
            file
        )

    with open(
        ADDRESS_STATS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        address_counts = json.load(
            file
        )

    print(
        f"Name token frequencies: "
        f"{len(name_counts):,}"
    )

    print(
        f"Address token frequencies: "
        f"{len(address_counts):,}"
    )

    # --------------------------------------------------------
    # Load Tree V3
    # --------------------------------------------------------

    print(
        "\nLoading Tree V3..."
    )

    model_bundle = joblib.load(
        MODEL_PATH
    )

    if not isinstance(
        model_bundle,
        dict,
    ):

        raise ValueError(
            "Expected Tree model file to contain "
            "a dictionary bundle."
        )

    if "model" not in model_bundle:

        raise ValueError(
            "Tree model bundle does not contain "
            "the 'model' key."
        )

    if "feature_columns" not in model_bundle:

        raise ValueError(
            "Tree model bundle does not contain "
            "the 'feature_columns' key."
        )

    model = model_bundle[
        "model"
    ]

    feature_columns = model_bundle[
        "feature_columns"
    ]

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Load candidate records
    # --------------------------------------------------------

    s2, s3 = load_candidate_data()

    # --------------------------------------------------------
    # Prepare output paths
    # --------------------------------------------------------

    candidate_output = Path(
        args.candidate_output
    )

    match_output = Path(
        args.match_output
    )

    candidate_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    match_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Start output files
    # --------------------------------------------------------

    processed = 0
    total_candidates = 0
    total_matches = 0

    with open(
        candidate_output,
        "w",
        encoding="utf-8",
        newline="",
    ) as candidate_file, open(
        match_output,
        "w",
        encoding="utf-8",
        newline="",
    ) as match_file:

        # Exact required headers.
        candidate_file.write(
            "source1_entity_id"
            "\t"
            "candidate_entity_ids\n"
        )

        match_file.write(
            "source1_entity_id"
            "\t"
            "matched_entity_ids\n"
        )

        # ----------------------------------------------------
        # Process Source 1 in batches
        # ----------------------------------------------------

        for s1_rows in iter_s1_batches(
            S1_PATH,
            S1_BATCH_SIZE,
        ):

            if args.s1_limit > 0:

                remaining = (
                    args.s1_limit
                    - processed
                )

                if remaining <= 0:
                    break

                s1_rows = s1_rows[
                    :remaining
                ]

            # ------------------------------------------------
            # Candidate retrieval
            # ------------------------------------------------

            with ThreadPoolExecutor(
                max_workers=WORKERS,
                initializer=initialize_worker,
                initargs=(
                    str(INDEX_PATH),
                    name_counts,
                    address_counts,
                ),
            ) as executor:

                candidate_lists = list(
                    executor.map(
                        fetch_top_candidates,
                        s1_rows,
                    )
                )

            # ------------------------------------------------
            # Matching
            # ------------------------------------------------

            for s1_row, candidate_ids in zip(
                s1_rows,
                candidate_lists,
            ):

                s1_id = safe_text(
                    s1_row["entity_id"]
                )

                candidate_ids = sorted(
                    set(
                        candidate_ids
                    )
                )

                total_candidates += len(
                    candidate_ids
                )

                # --------------------------------------------
                # candidate_pairs.tsv
                # --------------------------------------------

                candidate_file.write(
                    s1_id
                    + "\t"
                    + ",".join(
                        candidate_ids
                    )
                    + "\n"
                )

                # --------------------------------------------
                # Tree V3
                # --------------------------------------------

                matches = score_s1(
                    s1_row,
                    candidate_ids,
                    s2,
                    s3,
                    model,
                    feature_columns,
                )

                total_matches += len(
                    matches
                )

                # --------------------------------------------
                # matching_results.tsv
                # --------------------------------------------

                match_file.write(
                    s1_id
                    + "\t"
                    + ",".join(
                        matches
                    )
                    + "\n"
                )

                processed += 1

                if processed % 100 == 0:

                    mean_candidates = (
                        total_candidates
                        / processed
                    )

                    print(
                        f"Processed S1: "
                        f"{processed:,}"
                        f" | candidates: "
                        f"{total_candidates:,}"
                        f" | mean/S1: "
                        f"{mean_candidates:.2f}"
                        f" | matches: "
                        f"{total_matches:,}"
                    )

            if (
                args.s1_limit > 0
                and processed >= args.s1_limit
            ):

                break

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    mean_candidates = (
        total_candidates
        / processed
        if processed
        else 0.0
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL INFERENCE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"S1 processed       : "
        f"{processed:,}"
    )

    print(
        f"Candidate links    : "
        f"{total_candidates:,}"
    )

    print(
        f"Mean candidates/S1 : "
        f"{mean_candidates:.2f}"
    )

    print(
        f"Predicted matches  : "
        f"{total_matches:,}"
    )

    print(
        "\nCandidate output:"
    )

    print(
        candidate_output
    )

    print(
        "\nMatching output:"
    )

    print(
        match_output
    )


if __name__ == "__main__":
    main()