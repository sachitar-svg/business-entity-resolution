from pathlib import Path
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow.parquet as pq
import json

from generate_candidate_pairs import (
    build_block_terms,
    clean_country,
    safe_text,
    quote_fts_term,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE1_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "test_source1_normalized.parquet"
)

INDEX_PATH = (
    PROJECT_ROOT
    / "output"
    / "blocking_v2_test_index.sqlite"
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

SAMPLE_SIZE = 1000
TOP_K = 200
WORKERS = 8


# ============================================================
# WORKER GLOBALS
# ============================================================

WORKER_CONNECTION = None
WORKER_NAME_COUNTS = None
WORKER_ADDRESS_COUNTS = None


# ============================================================
# WORKER INITIALIZER
# ============================================================

def initialize_worker():

    global WORKER_CONNECTION
    global WORKER_NAME_COUNTS
    global WORKER_ADDRESS_COUNTS

    WORKER_CONNECTION = sqlite3.connect(
        INDEX_PATH
    )

    with open(
        NAME_STATS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        WORKER_NAME_COUNTS = json.load(
            file
        )

    with open(
        ADDRESS_STATS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        WORKER_ADDRESS_COUNTS = json.load(
            file
        )


# ============================================================
# CANDIDATE FETCH
# ============================================================

def fetch_candidates(
    terms,
    country,
):
    """
    Fetch TOP-K candidates for one S1 entity.
    """

    if not terms:
        return []

    query = " OR ".join(
        quote_fts_term(term)
        for term in sorted(terms)
    )

    rows = WORKER_CONNECTION.execute(
        """
        SELECT
            c.entity_id
        FROM block_fts AS f
        INNER JOIN candidate_records AS c
            ON c.rowid = f.rowid
        WHERE block_fts MATCH ?
          AND c.country = ?
        ORDER BY f.rank ASC, c.entity_id ASC
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
# WORKER TASK
# ============================================================

def process_row(row):

    s1_id = safe_text(
        row["entity_id"]
    )

    country = clean_country(
        row["country"]
    )

    terms = build_block_terms(
        safe_text(
            row["norm_name"]
        ),
        safe_text(
            row["compact_name"]
        ),
        safe_text(
            row["norm_address"]
        ),
        WORKER_NAME_COUNTS,
        WORKER_ADDRESS_COUNTS,
    )

    candidates = fetch_candidates(
        terms,
        country,
    )

    return (
        s1_id,
        len(candidates),
    )


# ============================================================
# LOAD SAMPLE
# ============================================================

def load_sample():

    parquet_file = pq.ParquetFile(
        SOURCE1_PATH
    )

    rows = []

    for batch in parquet_file.iter_batches(
        batch_size=SAMPLE_SIZE,
        columns=[
            "entity_id",
            "country",
            "norm_name",
            "compact_name",
            "norm_address",
        ],
    ):

        for row in batch.to_pylist():

            rows.append(row)

            if len(rows) >= SAMPLE_SIZE:

                return rows

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PARALLEL CANDIDATE GENERATION BENCHMARK")
    print("=" * 70)

    print(
        f"\nWorkers : {WORKERS}"
    )

    print(
        f"TOP-K   : {TOP_K}"
    )

    rows = load_sample()

    print(
        f"S1 sample: {len(rows):,}"
    )

    start_time = time.time()

    results = []

    # --------------------------------------------------------
    # Parallel processing
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=WORKERS,
        initializer=initialize_worker,
    ) as executor:

        futures = [
            executor.submit(
                process_row,
                row,
            )
            for row in rows
        ]

        for future in as_completed(
            futures
        ):

            results.append(
                future.result()
            )

    elapsed = (
        time.time()
        - start_time
    )

    total_candidates = sum(
        count
        for _, count in results
    )

    mean_candidates = (
        total_candidates
        / len(results)
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PARALLEL BENCHMARK COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"S1 processed          : "
        f"{len(results):,}"
    )

    print(
        f"Candidate links       : "
        f"{total_candidates:,}"
    )

    print(
        f"Mean candidates / S1  : "
        f"{mean_candidates:,.2f}"
    )

    print(
        f"Runtime                : "
        f"{elapsed:.1f}s"
    )

    print(
        f"Equivalent single-core: "
        f"{elapsed * WORKERS:.1f}s"
    )


if __name__ == "__main__":
    main()