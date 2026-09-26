from pathlib import Path
from collections import defaultdict
import argparse
import json
import sqlite3
import time

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

NORMALIZED_DIR = PROJECT_ROOT / "output" / "normalized"

SOURCE2_PATH = (
    NORMALIZED_DIR
    / "train_source2_normalized.parquet"
)

SOURCE3_PATH = (
    NORMALIZED_DIR
    / "train_source3_normalized.parquet"
)

SOURCE1_PATH = (
    NORMALIZED_DIR
    / "train_source1_normalized.parquet"
)

TOKEN_STATS_DIR = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "data"
    / "token_stats"
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

# This is the V2 threshold we actually benchmarked.
SINGLE_TOKEN_MAX_FREQ = 10_000

# Process Parquet files in chunks.
PARQUET_BATCH_SIZE = 100_000

# Number of rows inserted into SQLite per transaction.
SQLITE_BATCH_SIZE = 5_000

INDEX_VERSION = "v2_2026_09"


# ============================================================
# TEXT / TOKEN HELPERS
# ============================================================

def safe_text(value):
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_country(value):
    return safe_text(value).casefold()


def get_tokens(value):
    """
    Normalized text -> unique tokens.

    Tokens shorter than 2 characters are ignored,
    matching the project tokenization used by the
    teammate feature/blocking utilities.
    """

    value = safe_text(value)

    if not value:
        return set()

    return {
        token
        for token in value.split()
        if len(token) >= 2
    }


def rare_tokens(tokens, counts, max_frequency):
    """
    Return tokens whose global frequency <= max_frequency.
    """

    return {
        token
        for token in tokens
        if counts.get(token, 10**18) <= max_frequency
    }


def rarest_two_tokens(tokens, counts):
    """
    Select the two globally rarest tokens.

    Tie-breaking by token keeps the result deterministic.
    """

    if len(tokens) < 2:
        return ()

    ordered = sorted(
        tokens,
        key=lambda token: (
            counts.get(token, 10**18),
            token
        )
    )

    return tuple(ordered[:2])


def make_pair(tokens):
    """
    Canonical pair representation.
    """

    if len(tokens) != 2:
        return None

    return tuple(sorted(tokens))


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
    Build the exact V2 blocking channels.

    Channels:

    1. Exact normalized name
    2. Compact name
    3. Rare name tokens <= 10k
    4. Top-two / pair-based name key
    5. Rare address tokens <= 10k
    6. Top-two / pair-based address key

    Numeric blocking is intentionally NOT included.
    Relaxed 25k/50k rules are also intentionally NOT included.
    """

    terms = set()

    # --------------------------------------------------------
    # 1. Exact normalized name
    #
    # Convert spaces to underscores so the entire normalized
    # name becomes one FTS token.
    # --------------------------------------------------------

    if norm_name:
        exact_key = (
            "nexact_"
            + norm_name.replace(" ", "_")
        )

        terms.add(exact_key)

    # --------------------------------------------------------
    # 2. Compact name
    # --------------------------------------------------------

    if compact_name:

        compact_key = (
            "ncompact_"
            + compact_name
        )

        terms.add(compact_key)

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
        SINGLE_TOKEN_MAX_FREQ
    ):

        terms.add(
            "nrare_" + token
        )

    # --------------------------------------------------------
    # 4. Name pair
    # --------------------------------------------------------

    name_pair = make_pair(
        rarest_two_tokens(
            name_tokens,
            name_counts
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
        SINGLE_TOKEN_MAX_FREQ
    ):

        terms.add(
            "arare_" + token
        )

    # --------------------------------------------------------
    # 6. Address pair
    # --------------------------------------------------------

    address_pair = make_pair(
        rarest_two_tokens(
            address_tokens,
            address_counts
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
            "SQLite FTS5 is not available in this Python "
            "installation."
        ) from exc


def configure_sqlite(connection):

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

    for suffix in (
        "",
        "-wal",
        "-shm"
    ):

        path = Path(
            str(index_path) + suffix
        )

        if path.exists():
            path.unlink()


def build_index(
    index_path,
    name_counts,
    address_counts,
):

    print("\n" + "=" * 70)
    print("BUILDING V2 SQLITE / FTS5 INDEX")
    print("=" * 70)

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True
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
                SOURCE2_PATH,
                "Source 2"
            ),
            (
                SOURCE3_PATH,
                "Source 3"
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
                start=1
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
                        sorted(block_terms)
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
                str(SINGLE_TOKEN_MAX_FREQ),

            "source2_size":
                str(SOURCE2_PATH.stat().st_size),

            "source3_size":
                str(SOURCE3_PATH.stat().st_size),

            "source2_mtime":
                str(SOURCE2_PATH.stat().st_mtime_ns),

            "source3_mtime":
                str(SOURCE3_PATH.stat().st_mtime_ns),

            "name_stats_mtime":
                str(NAME_STATS_PATH.stat().st_mtime_ns),

            "address_stats_mtime":
                str(ADDRESS_STATS_PATH.stat().st_mtime_ns),

            "candidate_count":
                str(total_rows),
        }


        connection.executemany(
            """
            INSERT OR REPLACE INTO metadata(
                key,
                value
            )
            VALUES (?, ?)
            """,
            metadata.items()
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


def index_is_valid(index_path):

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
                str(SINGLE_TOKEN_MAX_FREQ),

            "source2_size":
                str(SOURCE2_PATH.stat().st_size),

            "source3_size":
                str(SOURCE3_PATH.stat().st_size),

            "source2_mtime":
                str(SOURCE2_PATH.stat().st_mtime_ns),

            "source3_mtime":
                str(SOURCE3_PATH.stat().st_mtime_ns),

            "name_stats_mtime":
                str(NAME_STATS_PATH.stat().st_mtime_ns),

            "address_stats_mtime":
                str(ADDRESS_STATS_PATH.stat().st_mtime_ns),
        }


        return all(
            rows.get(key) == value
            for key, value in required.items()
        )

    except (
        sqlite3.Error,
        OSError,
        KeyError
    ):

        return False


# ============================================================
# QUERY HELPERS
# ============================================================

def quote_fts_term(term):

    escaped = term.replace(
        '"',
        '""'
    )

    return f'"{escaped}"'


def build_query(
    norm_name,
    compact_name,
    norm_address,
    name_counts,
    address_counts,
):
    """
    Build the same six V2 query channels used by
    the Source-1 side.
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
):
    """
    Query FTS and apply the final V2 country filter.
    """

    if not match_query:
        return []

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
        )
    ).fetchall()

    return [
        row[0]
        for row in rows
    ]


# ============================================================
# SOURCE-1 PROCESSING
# ============================================================

def generate_candidates(
    index_path,
    output_path,
    name_counts,
    address_counts,
    s1_limit,
):

    connection = sqlite3.connect(
        str(index_path)
    )

    try:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        total_s1 = 0
        total_pairs = 0
        zero_candidate_s1 = 0

        start_time = time.time()


        # ----------------------------------------------------
        # We read Source 1 from normalized Parquet.
        # ----------------------------------------------------

        parquet_file = pq.ParquetFile(
            SOURCE1_PATH
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
                start=1
            ):

                chunk = batch.to_pandas()


                for row in chunk.itertuples(
                    index=False
                ):

                    if (
                        s1_limit > 0
                        and total_s1 >= s1_limit
                    ):
                        break


                    s1_id = safe_text(
                        row.entity_id
                    )

                    if not s1_id:
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


                    # ------------------------------------------------
                    # V2 query
                    # ------------------------------------------------

                    match_query = build_query(
                        norm_name,
                        compact_name,
                        norm_address,
                        name_counts,
                        address_counts,
                    )


                    candidates = fetch_candidates(
                        connection,
                        match_query,
                        country,
                    )


                    # Deterministic output.
                    candidates = sorted(
                        set(candidates)
                    )


                    if not candidates:
                        zero_candidate_s1 += 1


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


                if (
                    s1_limit > 0
                    and total_s1 >= s1_limit
                ):
                    break


        mean_candidates = (
            total_pairs / total_s1
            if total_s1
            else 0.0
        )


        print("\n" + "=" * 70)
        print("CANDIDATE GENERATION COMPLETE")
        print("=" * 70)

        print(
            f"S1 records processed      : "
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
            "Number of Source-1 rows to process. "
            "Use 0 for the full Source-1 dataset. "
            "Default: 1000 for safety."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Output candidate TSV path."
        ),
    )

    parser.add_argument(
        "--index",
        type=str,
        default=str(DEFAULT_INDEX_PATH),
        help=(
            "SQLite index path."
        ),
    )

    args = parser.parse_args()


    output_path = Path(
        args.output
    )

    index_path = Path(
        args.index
    )


    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    required_files = [
        SOURCE1_PATH,
        SOURCE2_PATH,
        SOURCE3_PATH,
        NAME_STATS_PATH,
        ADDRESS_STATS_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )


    print("=" * 70)
    print("V2 CANDIDATE GENERATION")
    print("=" * 70)

    print(
        f"Token threshold       : "
        f"{SINGLE_TOKEN_MAX_FREQ:,}"
    )

    print(
        f"S1 limit              : "
        f"{args.s1_limit:,}"
        if args.s1_limit > 0
        else
        "S1 limit              : FULL DATASET"
    )

    print(
        f"Index                 : "
        f"{index_path}"
    )

    print(
        f"Output                : "
        f"{output_path}"
    )


    # --------------------------------------------------------
    # Load token statistics
    # --------------------------------------------------------

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
        f"\nName token frequencies : "
        f"{len(name_counts):,}"
    )

    print(
        f"Address token frequencies: "
        f"{len(address_counts):,}"
    )


    # --------------------------------------------------------
    # Build/reuse index
    # --------------------------------------------------------

    if (
        args.rebuild_index
        or not index_is_valid(index_path)
    ):

        print(
            "\nNo compatible V2 index found."
        )

        build_index(
            index_path,
            name_counts,
            address_counts,
        )

    else:

        print(
            "\nReusing existing compatible V2 index."
        )


    # --------------------------------------------------------
    # Generate candidate pairs
    # --------------------------------------------------------

    generate_candidates(
        index_path,
        output_path,
        name_counts,
        address_counts,
        args.s1_limit,
    )


if __name__ == "__main__":
    main()