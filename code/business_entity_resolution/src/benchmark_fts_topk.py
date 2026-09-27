from pathlib import Path
import json
import sqlite3

import pandas as pd
import pyarrow.parquet as pq

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
    / "train_source1_normalized.parquet"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "train_ground_truth.parquet"
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

S1_IDS_PATH = (
    PROJECT_ROOT
    / "output"
    / "benchmark_relaxed_address_blocking.csv"
)

SAMPLE_SIZE = 1000

TOP_K_VALUES = [
    500,
    1000,
    2000,
    3000,
]


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
# LOAD BENCHMARK S1
# ============================================================

def load_s1_rows():

    ids_df = pd.read_csv(
        S1_IDS_PATH
    )

    if "s1_id" not in ids_df.columns:

        raise ValueError(
            "Benchmark ID file must contain "
            "an 's1_id' column."
        )

    target_ids = set(
        ids_df["s1_id"]
        .astype(str)
        .str.strip()
    )

    parquet_file = pq.ParquetFile(
        SOURCE1_PATH
    )

    rows = []

    for batch in parquet_file.iter_batches(
        batch_size=100_000,
        columns=[
            "entity_id",
            "country",
            "norm_name",
            "compact_name",
            "norm_address",
        ],
    ):

        for row in batch.to_pylist():

            s1_id = safe_text(
                row["entity_id"]
            )

            if s1_id in target_ids:

                rows.append(row)

                if len(rows) >= SAMPLE_SIZE:

                    return rows

    return rows


# ============================================================
# FETCH TOP-K CANDIDATES
# ============================================================

def fetch_top_k_candidates(
    connection,
    terms,
    country,
    top_k,
):
    """
    Retrieve only the strongest K blocking candidates.

    FTS5 BM25 score:
        lower score = stronger match
    """

    if not terms:
        return set()

    query = " OR ".join(
        quote_fts_term(term)
        for term in sorted(terms)
    )

    try:

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
                top_k,
            ),
        ).fetchall()

    except sqlite3.OperationalError as exc:

        raise RuntimeError(
            "SQLite FTS5 BM25 ranking failed. "
            "Check that this Python/SQLite build "
            "supports FTS5 ranking."
        ) from exc

    return {
        row[0]
        for row in rows
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FTS TOP-K BLOCKING BENCHMARK")
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
    # Load GT and S1 benchmark
    # --------------------------------------------------------

    ground_truth = load_ground_truth(
        GROUND_TRUTH_PATH
    )

    s1_rows = load_s1_rows()

    print(
        f"\nS1 sample: {len(s1_rows):,}"
    )

    # --------------------------------------------------------
    # Open existing training index
    # --------------------------------------------------------

    connection = sqlite3.connect(
        INDEX_PATH
    )

    try:

        for top_k in TOP_K_VALUES:

            candidate_links = 0
            true_matches = 0
            captured_matches = 0

            entities_with_gt = 0
            entities_100pct = 0
            entities_with_misses = 0

            print(
                "\n"
                + "-" * 70
            )

            print(
                f"Testing TOP-K = {top_k:,}"
            )

            for index, row in enumerate(
                s1_rows,
                start=1,
            ):

                s1_id = safe_text(
                    row["entity_id"]
                )

                country = clean_country(
                    row["country"]
                )

                # ------------------------------------------------
                # Build the same validated V2 blocking terms
                # ------------------------------------------------

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
                    name_counts,
                    address_counts,
                )

                # ------------------------------------------------
                # Retrieve only TOP-K candidates
                # ------------------------------------------------

                candidates = fetch_top_k_candidates(
                    connection,
                    terms,
                    country,
                    top_k,
                )

                gt_matches = ground_truth.get(
                    s1_id,
                    set(),
                )

                captured = (
                    gt_matches
                    & candidates
                )

                candidate_links += len(
                    candidates
                )

                true_matches += len(
                    gt_matches
                )

                captured_matches += len(
                    captured
                )

                if gt_matches:

                    entities_with_gt += 1

                    if captured == gt_matches:

                        entities_100pct += 1

                    else:

                        entities_with_misses += 1

                if index % 100 == 0:

                    print(
                        f"  Processed S1: "
                        f"{index:,}"
                    )

            # ----------------------------------------------------
            # Metrics
            # ----------------------------------------------------

            mean_candidates = (
                candidate_links
                / len(s1_rows)
                if s1_rows
                else 0.0
            )

            pair_recall = (
                captured_matches
                / true_matches
                if true_matches
                else 0.0
            )

            entity_recall = (
                entities_100pct
                / entities_with_gt
                if entities_with_gt
                else 0.0
            )

            print(
                f"\n  Mean candidates/S1 : "
                f"{mean_candidates:,.2f}"
            )

            print(
                f"  Candidate links     : "
                f"{candidate_links:,}"
            )

            print(
                f"  Pair recall         : "
                f"{pair_recall:.4%}"
            )

            print(
                f"  Entity recall       : "
                f"{entity_recall:.4%}"
            )

            print(
                f"  100% recall entities: "
                f"{entities_100pct:,}"
            )

            print(
                f"  Entities with misses: "
                f"{entities_with_misses:,}"
            )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "TOP-K BENCHMARK COMPLETE"
        )

        print(
            "=" * 70
        )

    finally:

        connection.close()


if __name__ == "__main__":
    main()