from pathlib import Path
import json
import sqlite3

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
    / "test_source1_normalized.parquet"
)

INDEX_PATH = (
    PROJECT_ROOT
    / "output"
    / "blocking_v2_test_index.sqlite"
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

SAMPLE_SIZE = 20


# ============================================================
# CHANNEL NAMES
# ============================================================

def get_channel(term):
    if term.startswith("nexact_"):
        return "exact_name"

    if term.startswith("ncompact_"):
        return "compact_name"

    if term.startswith("nrare_"):
        return "rare_name_token"

    if term.startswith("npair_"):
        return "name_pair"

    if term.startswith("arare_"):
        return "rare_address_token"

    if term.startswith("apair_"):
        return "address_pair"

    return "unknown"


# ============================================================
# FETCH COUNT
# ============================================================

def count_term_matches(
    connection,
    term,
    country,
):
    query = quote_fts_term(term)

    row = connection.execute(
        """
        SELECT COUNT(DISTINCT c.entity_id)
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
    ).fetchone()

    return int(row[0])


# ============================================================
# LOAD SOURCE 1 SAMPLE
# ============================================================

def load_sample_rows():

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
    print("TEST BLOCKING DIAGNOSTIC")
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

    connection = sqlite3.connect(
        INDEX_PATH
    )

    try:

        rows = load_sample_rows()

        print(
            f"\nSample S1 entities: {len(rows)}"
        )

        for row in rows:

            s1_id = safe_text(
                row["entity_id"]
            )

            country = clean_country(
                row["country"]
            )

            norm_name = safe_text(
                row["norm_name"]
            )

            compact_name = safe_text(
                row["compact_name"]
            )

            norm_address = safe_text(
                row["norm_address"]
            )

            terms = build_block_terms(
                norm_name,
                compact_name,
                norm_address,
                name_counts,
                address_counts,
            )

            channel_totals = {}

            term_details = []

            for term in sorted(terms):

                count = count_term_matches(
                    connection,
                    term,
                    country,
                )

                channel = get_channel(
                    term
                )

                channel_totals[channel] = (
                    channel_totals.get(
                        channel,
                        0
                    )
                    + count
                )

                term_details.append(
                    (
                        term,
                        channel,
                        count,
                    )
                )

            print("\n" + "-" * 70)
            print(
                f"S1: {s1_id}"
            )

            print(
                f"Country: {country}"
            )

            print("\nChannel counts:")

            for channel, count in sorted(
                channel_totals.items(),
                key=lambda item: item[1],
                reverse=True,
            ):

                print(
                    f"  {channel:<22} : "
                    f"{count:,}"
                )

            print("\nTop blocking terms:")

            for (
                term,
                channel,
                count,
            ) in sorted(
                term_details,
                key=lambda item: item[2],
                reverse=True,
            )[:10]:

                print(
                    f"  {channel:<22} "
                    f"{count:>10,}  "
                    f"{term}"
                )

    finally:

        connection.close()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DIAGNOSTIC COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()