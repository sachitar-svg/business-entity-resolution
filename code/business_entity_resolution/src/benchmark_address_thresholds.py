from pathlib import Path
import json
import sqlite3

import pandas as pd
import pyarrow.parquet as pq

from generate_candidate_pairs import (
    clean_country,
    safe_text,
    rare_tokens,
    rarest_two_tokens,
    make_pair,
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

NAME_MAX_FREQ = 10_000

ADDRESS_THRESHOLDS = [
    10_000,
    7_500,
    5_000,
    2_500,
    1_000,
]


# ============================================================
# BLOCKING TERMS
# ============================================================

def build_channel_terms(
    norm_name,
    compact_name,
    norm_address,
    name_counts,
    address_counts,
    address_max_freq,
):
    """
    Build V2 terms while varying only the
    rare-address-token frequency threshold.
    """

    channels = {
        "exact_name": set(),
        "compact_name": set(),
        "rare_name_token": set(),
        "name_pair": set(),
        "rare_address_token": set(),
        "address_pair": set(),
    }

    # --------------------------------------------------------
    # Exact normalized name
    # --------------------------------------------------------

    if norm_name:

        channels["exact_name"].add(
            "nexact_"
            + norm_name.replace(
                " ",
                "_",
            )
        )

    # --------------------------------------------------------
    # Compact name
    # --------------------------------------------------------

    if compact_name:

        channels["compact_name"].add(
            "ncompact_"
            + compact_name
        )

    # --------------------------------------------------------
    # Name tokens - fixed at 10k
    # --------------------------------------------------------

    name_tokens = {
        token
        for token in safe_text(
            norm_name
        ).split()
        if len(token) >= 2
    }

    for token in rare_tokens(
        name_tokens,
        name_counts,
        NAME_MAX_FREQ,
    ):

        channels["rare_name_token"].add(
            "nrare_"
            + token
        )

    name_pair = make_pair(
        rarest_two_tokens(
            name_tokens,
            name_counts,
        )
    )

    if name_pair:

        channels["name_pair"].add(
            "npair_"
            + name_pair[0]
            + "_"
            + name_pair[1]
        )

    # --------------------------------------------------------
    # Address tokens - variable threshold
    # --------------------------------------------------------

    address_tokens = {
        token
        for token in safe_text(
            norm_address
        ).split()
        if len(token) >= 2
    }

    for token in rare_tokens(
        address_tokens,
        address_counts,
        address_max_freq,
    ):

        channels["rare_address_token"].add(
            "arare_"
            + token
        )

    address_pair = make_pair(
        rarest_two_tokens(
            address_tokens,
            address_counts,
        )
    )

    if address_pair:

        channels["address_pair"].add(
            "apair_"
            + address_pair[0]
            + "_"
            + address_pair[1]
        )

    return channels


# ============================================================
# SQLITE QUERY
# ============================================================

def fetch_candidates(
    connection,
    terms,
    country,
):
    if not terms:
        return set()

    query = " OR ".join(
        quote_fts_term(term)
        for term in sorted(terms)
    )

    rows = connection.execute(
        """
        SELECT DISTINCT c.entity_id
        FROM block_fts AS f
        INNER JOIN candidate_records AS c
            ON c.rowid = f.rowid
        WHERE block_fts MATCH ?
          AND c.country = ?
        """,
        (
            query,
            country,
        ),
    ).fetchall()

    return {
        row[0]
        for row in rows
    }


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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ADDRESS TOKEN THRESHOLD BENCHMARK")
    print("=" * 70)

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

    ground_truth = load_ground_truth(
        GROUND_TRUTH_PATH
    )

    s1_rows = load_s1_rows()

    print(
        f"\nS1 sample: {len(s1_rows):,}"
    )

    connection = sqlite3.connect(
        INDEX_PATH
    )

    try:

        results = []

        for address_threshold in ADDRESS_THRESHOLDS:

            candidate_links = 0
            true_matches = 0
            captured_matches = 0

            entities_with_gt = 0
            entities_100pct = 0

            print(
                f"\nTesting address threshold: "
                f"{address_threshold:,}"
            )

            for row in s1_rows:

                s1_id = safe_text(
                    row["entity_id"]
                )

                country = clean_country(
                    row["country"]
                )

                channels = build_channel_terms(
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
                    address_threshold,
                )

                terms = set()

                for channel_terms in channels.values():

                    terms.update(
                        channel_terms
                    )

                candidates = fetch_candidates(
                    connection,
                    terms,
                    country,
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

            pair_recall = (
                captured_matches / true_matches
                if true_matches
                else 0.0
            )

            entity_recall = (
                entities_100pct / entities_with_gt
                if entities_with_gt
                else 0.0
            )

            mean_candidates = (
                candidate_links / len(s1_rows)
            )

            results.append(
                {
                    "address_threshold":
                        address_threshold,
                    "mean_candidates":
                        mean_candidates,
                    "candidate_links":
                        candidate_links,
                    "pair_recall":
                        pair_recall,
                    "entity_recall":
                        entity_recall,
                    "entities_100pct":
                        entities_100pct,
                }
            )

            print(
                f"  Mean candidates/S1 : "
                f"{mean_candidates:,.2f}"
            )

            print(
                f"  Pair recall         : "
                f"{pair_recall:.4%}"
            )

            print(
                f"  Entity recall       : "
                f"{entity_recall:.4%}"
            )

        # ----------------------------------------------------
        # Final table
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FINAL COMPARISON"
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

    finally:

        connection.close()


if __name__ == "__main__":
    main()