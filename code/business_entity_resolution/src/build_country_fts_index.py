from pathlib import Path
import json
import sqlite3
import time

import pandas as pd
import pyarrow.parquet as pq

from generate_candidate_pairs import (
    build_block_terms,
    clean_country,
    safe_text,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE2_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "test_source2_normalized.parquet"
)

SOURCE3_PATH = (
    PROJECT_ROOT
    / "output"
    / "normalized"
    / "test_source3_normalized.parquet"
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

PARQUET_BATCH_SIZE = 100_000
SQLITE_BATCH_SIZE = 5_000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING COUNTRY-AWARE TEST FTS INDEX")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    required = [
        SOURCE2_PATH,
        SOURCE3_PATH,
        INDEX_PATH,
        NAME_STATS_PATH,
        ADDRESS_STATS_PATH,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load token statistics
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
    # Open existing test index
    # --------------------------------------------------------

    connection = sqlite3.connect(
        str(INDEX_PATH)
    )

    try:

        print(
            "\nRemoving old country-aware table..."
        )

        connection.execute(
            "DROP TABLE IF EXISTS block_fts_country"
        )

        # ----------------------------------------------------
        # Create two-column FTS table
        #
        # country:
        #     used for filtering
        #
        # block_text:
        #     contains the exact existing V2 blocking terms
        #
        # bm25 weight for country will be 0
        # bm25 weight for block_text will be 1
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE VIRTUAL TABLE block_fts_country
            USING fts5(
                country,
                block_text,
                content=''
            )
            """
        )

        connection.commit()

        print(
            "Country-aware FTS table created."
        )

        next_rowid = 1
        total_rows = 0

        insert_rows = []

        start_time = time.time()

        # ----------------------------------------------------
        # Process Source 2 + Source 3
        #
        # IMPORTANT:
        # Row IDs must match candidate_records.rowid.
        # The original index inserted Source 2 first and
        # Source 3 second, skipping empty entity IDs.
        # ----------------------------------------------------

        for source_path, source_label in [
            (
                SOURCE2_PATH,
                "Source 2",
            ),
            (
                SOURCE3_PATH,
                "Source 3",
            ),
        ]:

            print(
                f"\nIndexing {source_label}..."
            )

            parquet_file = pq.ParquetFile(
                source_path
            )

            source_start = time.time()

            for batch_number, batch in enumerate(
                parquet_file.iter_batches(
                    batch_size=PARQUET_BATCH_SIZE,
                    columns=[
                        "entity_id",
                        "country",
                        "norm_name",
                        "compact_name",
                        "norm_address",
                    ],
                ),
                start=1,
            ):

                chunk = batch.to_pandas()

                for row in chunk.itertuples(
                    index=False
                ):

                    entity_id = safe_text(
                        row.entity_id
                    )

                    if not entity_id:

                        continue

                    country = clean_country(
                        row.country
                    )

                    norm_name = safe_text(
                        row.norm_name
                    )

                    compact_name = safe_text(
                        row.compact_name
                    )

                    norm_address = safe_text(
                        row.norm_address
                    )

                    terms = build_block_terms(
                        norm_name,
                        compact_name,
                        norm_address,
                        name_counts,
                        address_counts,
                    )

                    block_text = " ".join(
                        sorted(terms)
                    )

                    # Prefix country so the FTS token is
                    # unambiguous.
                    country_token = (
                        "country_"
                        + country
                        if country
                        else "country_missing"
                    )

                    insert_rows.append(
                        (
                            next_rowid,
                            country_token,
                            block_text,
                        )
                    )

                    next_rowid += 1
                    total_rows += 1

                    if (
                        len(insert_rows)
                        >= SQLITE_BATCH_SIZE
                    ):

                        connection.executemany(
                            """
                            INSERT INTO block_fts_country(
                                rowid,
                                country,
                                block_text
                            )
                            VALUES (?, ?, ?)
                            """,
                            insert_rows,
                        )

                        connection.commit()

                        insert_rows.clear()

                print(
                    f"  Batch {batch_number:>3}"
                    f" | indexed rows: "
                    f"{total_rows:,}"
                )

                del chunk

            print(
                f"{source_label} completed in "
                f"{time.time() - source_start:.1f}s"
            )

        # ----------------------------------------------------
        # Final batch
        # ----------------------------------------------------

        if insert_rows:

            connection.executemany(
                """
                INSERT INTO block_fts_country(
                    rowid,
                    country,
                    block_text
                )
                VALUES (?, ?, ?)
                """,
                insert_rows,
            )

            connection.commit()

        # ----------------------------------------------------
        # Optimize FTS
        # ----------------------------------------------------

        print(
            "\nOptimizing country-aware FTS..."
        )

        connection.execute(
            """
            INSERT INTO block_fts_country(
                block_fts_country
            )
            VALUES ('optimize')
            """
        )

        connection.commit()

        # ----------------------------------------------------
        # Store simple metadata
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS country_fts_metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            INSERT OR REPLACE INTO country_fts_metadata(
                key,
                value
            )
            VALUES (?, ?)
            """,
            (
                "row_count",
                str(total_rows),
            ),
        )

        connection.execute(
            """
            INSERT OR REPLACE INTO country_fts_metadata(
                key,
                value
            )
            VALUES (?, ?)
            """,
            (
                "version",
                "country_fts_v1",
            ),
        )

        connection.commit()

        print(
            "\n"
            + "=" * 70
        )

        print(
            "COUNTRY-AWARE FTS COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Rows indexed : {total_rows:,}"
        )

        print(
            f"Runtime      : "
            f"{time.time() - start_time:.1f}s"
        )

    finally:

        connection.close()


if __name__ == "__main__":
    main()