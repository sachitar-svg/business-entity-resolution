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
    SINGLE_TOKEN_MAX_FREQ,
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


# ============================================================
# BLOCKING CHANNELS
# ============================================================

def build_channel_terms(
    norm_name,
    compact_name,
    norm_address,
    name_counts,
    address_counts,
):
    """
    Build each V2 blocking channel separately.
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
    # Exact name
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
    # Name tokens
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
        SINGLE_TOKEN_MAX_FREQ,
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
    # Address tokens
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
        SINGLE_TOKEN_MAX_FREQ,
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
    """
    Fetch candidates for a set of blocking terms.
    """

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
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(path):

    gt = pd.read_parquet(path)

    required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required - set(gt.columns)

    if missing:

        raise ValueError(
            "Ground truth missing columns: "
            + ", ".join(sorted(missing))
        )

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

            if not text:

                matches = set()

            else:

                matches = {
                    value.strip()
                    for value in text.split(",")
                    if value.strip()
                }

        result[s1_id] = matches

    return result


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

def load_s1_rows():

    sample_ids = pd.read_csv(
        S1_IDS_PATH,
    )

    if "s1_id" not in sample_ids.columns:

        raise ValueError(
            "Benchmark ID file must contain "
            "an 's1_id' column."
        )

    target_ids = set(
        sample_ids["s1_id"]
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
    print("BLOCKING ABLATION BENCHMARK")
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

        # ----------------------------------------------------
        # Define candidate-channel variants
        # ----------------------------------------------------

        variants = {

            "current_v2": {
                "exact_name",
                "compact_name",
                "rare_name_token",
                "name_pair",
                "rare_address_token",
                "address_pair",
            },

            "no_rare_address": {
                "exact_name",
                "compact_name",
                "rare_name_token",
                "name_pair",
                "address_pair",
            },

            "name_only": {
                "exact_name",
                "compact_name",
                "rare_name_token",
                "name_pair",
            },

            "pairs_plus_exact": {
                "exact_name",
                "compact_name",
                "name_pair",
                "address_pair",
            },

            "no_single_tokens": {
                "exact_name",
                "compact_name",
                "name_pair",
                "address_pair",
            },
        }

        stats = {
            name: {
                "candidate_links": 0,
                "true_matches": 0,
                "captured_matches": 0,
                "entities_with_gt": 0,
                "entities_100pct_recall": 0,
                "entities_with_misses": 0,
            }
            for name in variants
        }

        # ----------------------------------------------------
        # Evaluate each S1
        # ----------------------------------------------------

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
            )

            true_matches = ground_truth.get(
                s1_id,
                set(),
            )

            for variant_name, enabled_channels in variants.items():

                terms = set()

                for channel in enabled_channels:

                    terms.update(
                        channels[channel]
                    )

                candidates = fetch_candidates(
                    connection,
                    terms,
                    country,
                )

                captured = (
                    true_matches
                    & candidates
                )

                stats[variant_name][
                    "candidate_links"
                ] += len(candidates)

                stats[variant_name][
                    "true_matches"
                ] += len(true_matches)

                stats[variant_name][
                    "captured_matches"
                ] += len(captured)

                if true_matches:

                    stats[variant_name][
                        "entities_with_gt"
                    ] += 1

                    if captured == true_matches:

                        stats[variant_name][
                            "entities_100pct_recall"
                        ] += 1

                    else:

                        stats[variant_name][
                            "entities_with_misses"
                        ] += 1

            if index % 100 == 0:

                print(
                    f"Processed S1: {index:,}"
                )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "RESULTS"
        )

        print(
            "=" * 70
        )

        total_s1 = len(s1_rows)

        for variant_name, data in variants.items():

            values = stats[variant_name]

            mean_candidates = (
                values["candidate_links"]
                / total_s1
                if total_s1
                else 0.0
            )

            pair_recall = (
                values["captured_matches"]
                / values["true_matches"]
                if values["true_matches"]
                else 0.0
            )

            entity_recall = (
                values["entities_100pct_recall"]
                / values["entities_with_gt"]
                if values["entities_with_gt"]
                else 0.0
            )

            print(
                f"\n{variant_name}"
            )

            print(
                f"  Mean candidates / S1 : "
                f"{mean_candidates:,.2f}"
            )

            print(
                f"  Candidate links       : "
                f"{values['candidate_links']:,}"
            )

            print(
                f"  Pair recall           : "
                f"{pair_recall:.4%}"
            )

            print(
                f"  Entity recall         : "
                f"{entity_recall:.4%}"
            )

            print(
                f"  100% recall entities  : "
                f"{values['entities_100pct_recall']:,}"
            )

            print(
                f"  Entities with misses  : "
                f"{values['entities_with_misses']:,}"
            )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "BENCHMARK COMPLETE"
        )

        print(
            "=" * 70
        )

    finally:

        connection.close()


if __name__ == "__main__":
    main()