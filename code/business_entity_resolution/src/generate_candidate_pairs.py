from pathlib import Path
import argparse
import csv
import json
import sqlite3
import time

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

NORMALIZED_DIR = (
    PROJECT_ROOT
    / "output"
    / "normalized"
)

TOKEN_STATS_DIR = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "data"
    / "token_stats"
)


# ============================================================
# DEFAULT TRAIN PATHS
# ============================================================

SOURCE1_PATH = (
    NORMALIZED_DIR
    / "train_source1_normalized.parquet"
)

SOURCE2_PATH = (
    NORMALIZED_DIR
    / "train_source2_normalized.parquet"
)

SOURCE3_PATH = (
    NORMALIZED_DIR
    / "train_source3_normalized.parquet"
)

NAME_STATS_PATH = (
    TOKEN_STATS_DIR
    / "name_token_counts.json"
)

ADDRESS_STATS_PATH = (
    TOKEN_STATS_DIR
    / "address_token_counts.json"
)

DEFAULT_INDEX_PATH = (
    PROJECT_ROOT
    / "output"
    / "blocking_v2_index.sqlite"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "candidate_pairs.tsv"
)


# ============================================================
# V2 CONFIGURATION
# ============================================================

SINGLE_TOKEN_MAX_FREQ = 10_000

PARQUET_BATCH_SIZE = 100_000

SQLITE_BATCH_SIZE = 5_000

INDEX_VERSION = "v2_2026_09"


# ============================================================
# TEXT / TOKEN HELPERS
# ============================================================

def safe_text(value):
    """
    Convert a value into clean string text.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_country(value):
    """
    Normalize country value for comparison.
    """

    return safe_text(value).casefold()


def get_tokens(value):
    """
    Convert normalized text into unique tokens.

    Tokens shorter than 2 characters are ignored.
    """

    value = safe_text(value)

    if not value:
        return set()

    return {
        token
        for token in value.split()
        if len(token) >= 2
    }


def rare_tokens(
    tokens,
    counts,
    max_frequency,
):
    """
    Return tokens whose global frequency is <= max_frequency.
    """

    return {
        token
        for token in tokens
        if counts.get(
            token,
            10**18
        ) <= max_frequency
    }


def rarest_two_tokens(
    tokens,
    counts,
):
    """
    Select the two globally rarest tokens.

    Tie-breaking by token keeps the result deterministic.
    """

    if len(tokens) < 2:
        return ()

    ordered = sorted(
        tokens,
        key=lambda token: (
            counts.get(
                token,
                10**18
            ),
            token,
        ),
    )

    return tuple(
        ordered[:2]
    )


def make_pair(tokens):
    """
    Create a canonical pair representation.
    """

    if len(tokens) != 2:
        return None

    return tuple(
        sorted(tokens)
    )


# ============================================================
# BLOCKING KEY BUILDING
# ============================================================

def build_block_terms(
    norm_name,
    compact_name,
    norm_address,
    name_counts,
    address_counts,
):
    """
    Build the exact validated V2 blocking channels.

    Channels:

    1. Exact normalized name
    2. Compact name
    3. Rare name tokens <= 10k
    4. Top-two / pair-based name key
    5. Rare address tokens <= 10k
    6. Top-two / pair-based address key

    Numeric blocking is intentionally NOT included.
    """

    terms = set()

    # --------------------------------------------------------
    # 1. Exact normalized name
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
    # 2. Compact name
    # --------------------------------------------------------

    if compact_name:

        terms.add(
            "ncompact_"
            + compact_name
        )

    # --------------------------------------------------------
    # Name tokens
    # --------------------------------------------------------

    name_tokens = get_tokens(
        norm_name
    )

    # --------------------------------------------------------
    # 3. Rare name tokens <= 10k
    # --------------------------------------------------------

    for token in rare_tokens(
        name_tokens,
        name_counts,
        SINGLE_TOKEN_MAX_FREQ,
    ):

        terms.add(
            "nrare_"
            + token
        )

    # --------------------------------------------------------
    # 4. Name pair
    # --------------------------------------------------------

    name_pair = make_pair(
        rarest_two_tokens(
            name_tokens,
            name_counts,
        )
    )

    if name_pair:

        terms.add(
            "npair_"
            + name_pair[0]
            + "_"
            + name_pair[1]
        )

    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    address_tokens = get_tokens(
        norm_address
    )

    # --------------------------------------------------------
    # 5. Rare address tokens <= 10k
    # --------------------------------------------------------

    for token in rare_tokens(
        address_tokens,
        address_counts,
        SINGLE_TOKEN_MAX_FREQ,
    ):

        terms.add(
            "arare_"
            + token
        )

    # --------------------------------------------------------
    # 6. Address pair
    # --------------------------------------------------------

    address_pair = make_pair(
        rarest_two_tokens(
            address_tokens,
            address_counts,
        )
    )

    if address_pair:

        terms.add(
            "apair_"
            + address_pair[0]
            + "_"
            + address_pair[1]
        )

    return terms


# ============================================================
# SQLITE INDEX
# ============================================================

def check_fts5(connection):
    """
    Confirm SQLite has FTS5 support.
    """

    try:

        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS "
            "fts5_test USING fts5(x)"
        )

        connection.execute(
            "DROP TABLE IF EXISTS fts5_test"
        )

    except sqlite3.OperationalError as exc:

        raise RuntimeError(
            "SQLite FTS5 is not available in this "
            "Python installation."
        ) from exc


def configure_sqlite(connection):
    """
    Configure SQLite for this indexing workload.
    """

    connection.execute(
        "PRAGMA journal_mode=WAL"
    )

    connection.execute(
        "PRAGMA synchronous=NORMAL"
    )

    connection.execute(
        "PRAGMA temp_store=MEMORY"
    )

    connection.execute(
        "PRAGMA cache_size=-100000"
    )

    connection.execute(
        "PRAGMA mmap_size=268435456"
    )


def create_schema(connection):
    """
    Create all required SQLite tables.
    """

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_records (
            rowid INTEGER PRIMARY KEY,
            entity_id TEXT NOT NULL UNIQUE,
            country TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS block_fts
        USING fts5(
            block_text,
            content=''
        )
        """
    )

    connection.commit()


def remove_old_index(index_path):
    """
    Remove an existing SQLite database and its WAL files.
    """

    for suffix in (
        "",
        "-wal",
        "-shm",
    ):

        path = Path(
            str(index_path)
            + suffix
        )

        if path.exists():
            path.unlink()


def build_index(
    index_path,
    name_counts,
    address_counts,
    source2_path,
    source3_path,
    name_stats_path,
    address_stats_path,
):
    """
    Build the disk-backed SQLite/FTS5 V2 index.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BUILDING V2 SQLITE / FTS5 INDEX"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSource 2: {source2_path}"
    )

    print(
        f"Source 3: {source3_path}"
    )

    print(
        f"Name stats: {name_stats_path}"
    )

    print(
        f"Address stats: {address_stats_path}"
    )

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    remove_old_index(
        index_path
    )

    connection = sqlite3.connect(
        str(index_path)
    )

    try:

        configure_sqlite(
            connection
        )

        check_fts5(
            connection
        )

        create_schema(
            connection
        )

        next_rowid = 1
        total_rows = 0

        insert_records = []
        insert_fts = []

        overall_start = time.time()

        for source_path, source_label in [
            (
                source2_path,
                "Source 2",
            ),
            (
                source3_path,
                "Source 3",
            ),
        ]:

            print(
                f"\nIndexing {source_label}:"
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

                    block_terms = build_block_terms(
                        norm_name,
                        compact_name,
                        norm_address,
                        name_counts,
                        address_counts,
                    )

                    block_text = " ".join(
                        sorted(
                            block_terms
                        )
                    )

                    insert_records.append(
                        (
                            next_rowid,
                            entity_id,
                            country,
                        )
                    )

                    insert_fts.append(
                        (
                            next_rowid,
                            block_text,
                        )
                    )

                    next_rowid += 1
                    total_rows += 1

                    if (
                        len(insert_records)
                        >= SQLITE_BATCH_SIZE
                    ):

                        connection.executemany(
                            """
                            INSERT INTO candidate_records(
                                rowid,
                                entity_id,
                                country
                            )
                            VALUES (?, ?, ?)
                            """,
                            insert_records,
                        )

                        connection.executemany(
                            """
                            INSERT INTO block_fts(
                                rowid,
                                block_text
                            )
                            VALUES (?, ?)
                            """,
                            insert_fts,
                        )

                        connection.commit()

                        insert_records.clear()
                        insert_fts.clear()

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
        # Flush final batch
        # ----------------------------------------------------

        if insert_records:

            connection.executemany(
                """
                INSERT INTO candidate_records(
                    rowid,
                    entity_id,
                    country
                )
                VALUES (?, ?, ?)
                """,
                insert_records,
            )

            connection.executemany(
                """
                INSERT INTO block_fts(
                    rowid,
                    block_text
                )
                VALUES (?, ?)
                """,
                insert_fts,
            )

            connection.commit()

        # ----------------------------------------------------
        # Optimize FTS
        # ----------------------------------------------------

        print(
            "\nOptimizing FTS index..."
        )

        connection.execute(
            """
            INSERT INTO block_fts(block_fts)
            VALUES ('optimize')
            """
        )

        connection.commit()

        # ----------------------------------------------------
        # Store metadata
        # ----------------------------------------------------

        metadata = {

            "index_version":
                INDEX_VERSION,

            "single_token_max_freq":
                str(
                    SINGLE_TOKEN_MAX_FREQ
                ),

            "source2_size":
                str(
                    source2_path.stat().st_size
                ),

            "source3_size":
                str(
                    source3_path.stat().st_size
                ),

            "source2_mtime":
                str(
                    source2_path.stat().st_mtime_ns
                ),

            "source3_mtime":
                str(
                    source3_path.stat().st_mtime_ns
                ),

            "name_stats_mtime":
                str(
                    name_stats_path.stat().st_mtime_ns
                ),

            "address_stats_mtime":
                str(
                    address_stats_path.stat().st_mtime_ns
                ),

            "candidate_count":
                str(
                    total_rows
                ),
        }

        connection.executemany(
            """
            INSERT OR REPLACE INTO metadata(
                key,
                value
            )
            VALUES (?, ?)
            """,
            metadata.items(),
        )

        connection.commit()

        print(
            "\nTotal candidate records indexed: "
            f"{total_rows:,}"
        )

        print(
            "Index build time: "
            f"{time.time() - overall_start:.1f}s"
        )

    finally:

        connection.close()


def index_is_valid(
    index_path,
    source2_path,
    source3_path,
    name_stats_path,
    address_stats_path,
):
    """
    Check whether the existing SQLite index matches
    the current V2 configuration, source files, and
    token-statistics files.
    """

    if not index_path.exists():
        return False

    try:

        connection = sqlite3.connect(
            str(index_path)
        )

        try:

            rows = dict(
                connection.execute(
                    "SELECT key, value FROM metadata"
                ).fetchall()
            )

        finally:

            connection.close()

        required = {

            "index_version":
                INDEX_VERSION,

            "single_token_max_freq":
                str(
                    SINGLE_TOKEN_MAX_FREQ
                ),

            "source2_size":
                str(
                    source2_path.stat().st_size
                ),

            "source3_size":
                str(
                    source3_path.stat().st_size
                ),

            "source2_mtime":
                str(
                    source2_path.stat().st_mtime_ns
                ),

            "source3_mtime":
                str(
                    source3_path.stat().st_mtime_ns
                ),

            "name_stats_mtime":
                str(
                    name_stats_path.stat().st_mtime_ns
                ),

            "address_stats_mtime":
                str(
                    address_stats_path.stat().st_mtime_ns
                ),
        }

        return all(
            rows.get(key) == value
            for key, value in required.items()
        )

    except (
        sqlite3.Error,
        OSError,
        KeyError,
    ):

        return False


# ============================================================
# QUERY HELPERS
# ============================================================

def quote_fts_term(term):
    """
    Quote an FTS5 search term safely.
    """

    escaped = term.replace(
        '"',
        '""'
    )

    return (
        f'"{escaped}"'
    )


def build_query(
    norm_name,
    compact_name,
    norm_address,
    name_counts,
    address_counts,
):
    """
    Build the same six V2 query channels.
    """

    terms = build_block_terms(
        norm_name,
        compact_name,
        norm_address,
        name_counts,
        address_counts,
    )

    if not terms:
        return ""

    return " OR ".join(
        quote_fts_term(term)
        for term in sorted(terms)
    )


def fetch_candidates(
    connection,
    match_query,
    country,
    top_k=None,
):
    """
    Query V2 candidates.

    For TOP-K mode, use the country-aware FTS table.
    Country filtering happens inside FTS.

    BM25 scoring uses:
        country weight   = 0
        block_text weight = 1

    Therefore country is a filter only and does not
    contribute to the ranking score.
    """

    if not match_query:
        return []

    # --------------------------------------------------------
    # TOP-K mode
    # --------------------------------------------------------

    if top_k is not None:

        country = clean_country(
            country
        )

        if country:

            country_token = (
                "country_"
                + country
            )

        else:

            country_token = (
                "country_missing"
            )

        ranked_query = (
            '"'
            + country_token
            + '" AND ('
            + match_query
            + ')'
        )

        rows = connection.execute(
            """
            SELECT
                c.entity_id
            FROM block_fts_country AS f
            INNER JOIN candidate_records AS c
                ON c.rowid = f.rowid
            WHERE block_fts_country MATCH ?
            ORDER BY
                bm25(
                    block_fts_country,
                    0.0,
                    1.0
                ) ASC,
                c.entity_id ASC
            LIMIT ?
            """,
            (
                ranked_query,
                top_k,
            ),
        ).fetchall()

        return [
            row[0]
            for row in rows
        ]

    # --------------------------------------------------------
    # Full V2 mode
    # --------------------------------------------------------

    rows = connection.execute(
        """
        SELECT DISTINCT
            c.entity_id
        FROM block_fts AS f
        INNER JOIN candidate_records AS c
            ON c.rowid = f.rowid
        WHERE block_fts MATCH ?
          AND c.country = ?
        """,
        (
            match_query,
            country,
        ),
    ).fetchall()

    return [
        row[0]
        for row in rows
    ]
  


# ============================================================
# EXACT S1 ID LOADING
# ============================================================

def load_s1_ids(ids_path):
    """
    Load exact Source-1 entity IDs from a CSV file.

    Expected column:
        s1_id
    """

    ids = []

    with open(
        ids_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        if not reader.fieldnames:

            raise ValueError(
                "The S1 IDs file has no header."
            )

        if "s1_id" not in reader.fieldnames:

            raise ValueError(
                "S1 IDs file must contain a "
                "column named 's1_id'."
            )

        for row in reader:

            s1_id = safe_text(
                row.get(
                    "s1_id"
                )
            )

            if s1_id:

                ids.append(
                    s1_id
                )

    if not ids:

        raise ValueError(
            "No valid S1 IDs were found in "
            "the IDs file."
        )

    return ids


# ============================================================
# SOURCE-1 PROCESSING
# ============================================================

def generate_candidates(
    index_path,
    output_path,
    name_counts,
    address_counts,
    source1_path,
    s1_limit,
    s1_ids=None,
    top_k=None,
):
    """
    Generate candidate entity IDs for Source 1.

    Two S1 selection modes:

    1. Normal mode:
       Process the first --s1-limit rows.

    2. Exact-ID mode:
       Process ONLY the supplied S1 IDs.

    Candidate retrieval modes:

    1. Full mode:
       Return all V2 candidates.

    2. TOP-K mode:
       Rank candidates using FTS5 BM25 and return
       only the strongest K candidates.
    """

    connection = sqlite3.connect(
        str(index_path)
    )

    try:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        total_s1 = 0
        total_pairs = 0
        zero_candidate_s1 = 0

        # ----------------------------------------------------
        # Exact Source-1 ID mode
        # ----------------------------------------------------

        if s1_ids is not None:

            target_s1_ids = set(
                s1_ids
            )

            if not target_s1_ids:

                raise ValueError(
                    "The exact S1 ID list is empty."
                )

        else:

            target_s1_ids = None

        processed_s1_ids = set()

        start_time = time.time()

        # ----------------------------------------------------
        # Read Source 1 from normalized Parquet
        # ----------------------------------------------------

        parquet_file = pq.ParquetFile(
            source1_path
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:

            output_file.write(
                "source1_entity_id"
                "\t"
                "candidate_entity_ids\n"
            )

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

                    s1_id = safe_text(
                        row.entity_id
                    )

                    if not s1_id:
                        continue

                    # ------------------------------------------------
                    # EXACT-ID MODE
                    # ------------------------------------------------

                    if target_s1_ids is not None:

                        if s1_id not in target_s1_ids:
                            continue

                        if s1_id in processed_s1_ids:
                            continue

                    # ------------------------------------------------
                    # NORMAL LIMIT MODE
                    # ------------------------------------------------

                    else:

                        if (
                            s1_limit > 0
                            and total_s1 >= s1_limit
                        ):
                            break

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

                    # ------------------------------------------------
                    # Build V2 query
                    # ------------------------------------------------

                    match_query = build_query(
                        norm_name,
                        compact_name,
                        norm_address,
                        name_counts,
                        address_counts,
                    )

                    # ------------------------------------------------
                    # Fetch candidates
                    # ------------------------------------------------

                    candidates = fetch_candidates(
                        connection,
                        match_query,
                        country,
                        top_k,
                    )

                    # ------------------------------------------------
                    # Deterministic output
                    # ------------------------------------------------

                    candidates = sorted(
                        set(candidates)
                    )

                    if not candidates:

                        zero_candidate_s1 += 1

                    # ------------------------------------------------
                    # Write required candidate format
                    # ------------------------------------------------

                    output_file.write(
                        s1_id
                        + "\t"
                        + ",".join(candidates)
                        + "\n"
                    )

                    total_pairs += len(
                        candidates
                    )

                    total_s1 += 1

                    # ------------------------------------------------
                    # Track processed exact IDs
                    # ------------------------------------------------

                    if target_s1_ids is not None:

                        processed_s1_ids.add(
                            s1_id
                        )

                    # ------------------------------------------------
                    # Progress
                    # ------------------------------------------------

                    if total_s1 % 100 == 0:

                        mean_candidates = (
                            total_pairs
                            / total_s1
                        )

                        print(
                            f"Processed S1: "
                            f"{total_s1:,}"
                            f" | pairs: "
                            f"{total_pairs:,}"
                            f" | mean candidates/S1: "
                            f"{mean_candidates:,.2f}"
                        )

                del chunk

                # ----------------------------------------------------
                # STOP AFTER ALL EXACT IDS ARE FOUND
                # ----------------------------------------------------

                if (
                    target_s1_ids is not None
                    and processed_s1_ids
                    >= target_s1_ids
                ):
                    break

                # ----------------------------------------------------
                # NORMAL LIMIT STOP
                # ----------------------------------------------------

                if (
                    target_s1_ids is None
                    and s1_limit > 0
                    and total_s1 >= s1_limit
                ):
                    break

        # --------------------------------------------------------
        # Verify exact-ID coverage
        # --------------------------------------------------------

        if target_s1_ids is not None:

            missing_ids = (
                target_s1_ids
                - processed_s1_ids
            )

            if missing_ids:

                print(
                    "\nWARNING:"
                )

                print(
                    f"Requested exact S1 IDs : "
                    f"{len(target_s1_ids):,}"
                )

                print(
                    f"Processed exact S1 IDs : "
                    f"{len(processed_s1_ids):,}"
                )

                print(
                    f"Missing S1 IDs          : "
                    f"{len(missing_ids):,}"
                )

                print(
                    "\nFirst missing IDs:"
                )

                for missing_id in sorted(
                    missing_ids
                )[:10]:

                    print(
                        f"  {missing_id}"
                    )

            else:

                print(
                    "\nAll requested exact S1 IDs "
                    "were processed successfully."
                )

        # --------------------------------------------------------
        # Final statistics
        # --------------------------------------------------------

        mean_candidates = (
            total_pairs / total_s1
            if total_s1
            else 0.0
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "CANDIDATE GENERATION COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"S1 records processed       : "
            f"{total_s1:,}"
        )

        print(
            f"Candidate links written    : "
            f"{total_pairs:,}"
        )

        print(
            f"Mean candidates / S1       : "
            f"{mean_candidates:,.2f}"
        )

        print(
            f"S1 with zero candidates    : "
            f"{zero_candidate_s1:,}"
        )

        if top_k is None:

            print(
                "Candidate retrieval        : "
                "FULL V2"
            )

        else:

            print(
                f"Candidate retrieval        : "
                f"TOP-{top_k:,}"
            )

        print(
            f"Runtime                    : "
            f"{time.time() - start_time:.1f}s"
        )

        print(
            "\nOutput:"
        )

        print(
            output_path
        )

    finally:

        connection.close()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Scalable V2 candidate generation for "
            "business entity resolution."
        )
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help=(
            "Rebuild the disk-backed V2 SQLite/FTS5 index."
        ),
    )

    parser.add_argument(
        "--s1-limit",
        type=int,
        default=1000,
        help=(
            "Number of Source-1 rows to process when "
            "--s1-ids-file is not supplied. "
            "Use 0 for the full Source-1 dataset. "
            "Default: 1000 for safety."
        ),
    )

    parser.add_argument(
        "--s1-ids-file",
        type=str,
        default=None,
        help=(
            "CSV file containing exact Source-1 IDs "
            "in a column named 's1_id'. "
            "When supplied, this takes priority "
            "over --s1-limit."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DEFAULT_OUTPUT_PATH
        ),
        help=(
            "Output candidate TSV path."
        ),
    )

    parser.add_argument(
        "--index",
        type=str,
        default=str(
            DEFAULT_INDEX_PATH
        ),
        help=(
            "SQLite index path."
        ),
    )

    parser.add_argument(
        "--source1",
        type=str,
        default=str(
            SOURCE1_PATH
        ),
        help=(
            "Source 1 normalized Parquet file."
        ),
    )

    parser.add_argument(
        "--source2",
        type=str,
        default=str(
            SOURCE2_PATH
        ),
        help=(
            "Source 2 normalized Parquet file."
        ),
    )

    parser.add_argument(
        "--source3",
        type=str,
        default=str(
            SOURCE3_PATH
        ),
        help=(
            "Source 3 normalized Parquet file."
        ),
    )

    parser.add_argument(
        "--name-stats",
        type=str,
        default=str(
            NAME_STATS_PATH
        ),
        help=(
            "Name token statistics JSON file."
        ),
    )

    parser.add_argument(
        "--address-stats",
        type=str,
        default=str(
            ADDRESS_STATS_PATH
        ),
        help=(
            "Address token statistics JSON file."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Keep only the top K candidates per S1 "
            "after FTS5 BM25 ranking. "
            "Omit this option for full V2 candidates."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate TOP-K
    # --------------------------------------------------------

    if args.top_k is not None:

        if args.top_k <= 0:

            raise ValueError(
                "--top-k must be greater than 0."
            )

    # --------------------------------------------------------
    # Convert paths
    # --------------------------------------------------------

    output_path = Path(
        args.output
    )

    index_path = Path(
        args.index
    )

    source1_path = Path(
        args.source1
    )

    source2_path = Path(
        args.source2
    )

    source3_path = Path(
        args.source3
    )

    name_stats_path = Path(
        args.name_stats
    )

    address_stats_path = Path(
        args.address_stats
    )

    # --------------------------------------------------------
    # Load exact S1 IDs if supplied
    # --------------------------------------------------------

    s1_ids = None

    if args.s1_ids_file:

        s1_ids = load_s1_ids(
            Path(
                args.s1_ids_file
            )
        )

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    required_files = [
        source1_path,
        source2_path,
        source3_path,
        name_stats_path,
        address_stats_path,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "V2 CANDIDATE GENERATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Source 1               : "
        f"{source1_path}"
    )

    print(
        f"Source 2               : "
        f"{source2_path}"
    )

    print(
        f"Source 3               : "
        f"{source3_path}"
    )

    print(
        f"Name stats             : "
        f"{name_stats_path}"
    )

    print(
        f"Address stats          : "
        f"{address_stats_path}"
    )

    print(
        f"Token threshold        : "
        f"{SINGLE_TOKEN_MAX_FREQ:,}"
    )

    if args.top_k is None:

        print(
            "Candidate retrieval    : FULL V2"
        )

    else:

        print(
            f"Candidate retrieval    : "
            f"TOP-{args.top_k:,}"
        )

    if s1_ids is not None:

        print(
            f"Exact S1 IDs loaded    : "
            f"{len(s1_ids):,}"
        )

        print(
            "S1 limit               : "
            "IGNORED (exact ID mode)"
        )

    else:

        print(
            f"S1 limit               : "
            f"{args.s1_limit:,}"
            if args.s1_limit > 0
            else
            "S1 limit               : FULL DATASET"
        )

    print(
        f"Index                  : "
        f"{index_path}"
    )

    print(
        f"Output                 : "
        f"{output_path}"
    )

    # --------------------------------------------------------
    # Load token statistics
    # --------------------------------------------------------

    with open(
        name_stats_path,
        "r",
        encoding="utf-8",
    ) as file:

        name_counts = json.load(
            file
        )

    with open(
        address_stats_path,
        "r",
        encoding="utf-8",
    ) as file:

        address_counts = json.load(
            file
        )

    print(
        f"\nName token frequencies : "
        f"{len(name_counts):,}"
    )

    print(
        f"Address token frequencies: "
        f"{len(address_counts):,}"
    )

    # --------------------------------------------------------
    # Build or reuse SQLite index
    # --------------------------------------------------------

    if (
        args.rebuild_index
        or not index_is_valid(
            index_path,
            source2_path,
            source3_path,
            name_stats_path,
            address_stats_path,
        )
    ):

        print(
            "\nNo compatible V2 index found."
        )

        build_index(
            index_path,
            name_counts,
            address_counts,
            source2_path,
            source3_path,
            name_stats_path,
            address_stats_path,
        )

    else:

        print(
            "\nReusing existing compatible V2 index."
        )

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    generate_candidates(
        index_path,
        output_path,
        name_counts,
        address_counts,
        source1_path,
        args.s1_limit,
        s1_ids,
        args.top_k,
    )


if __name__ == "__main__":
    main()