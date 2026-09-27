from pathlib import Path
from collections import Counter
import json
import time

import pyarrow.parquet as pq


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

NORMALIZED_DIR = (
    PROJECT_ROOT
    / "output"
    / "normalized"
)

SOURCE2_PATH = (
    NORMALIZED_DIR
    / "test_source2_normalized.parquet"
)

SOURCE3_PATH = (
    NORMALIZED_DIR
    / "test_source3_normalized.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "test_token_stats"
)

NAME_OUTPUT = (
    OUTPUT_DIR
    / "name_token_counts.json"
)

ADDRESS_OUTPUT = (
    OUTPUT_DIR
    / "address_token_counts.json"
)

BATCH_SIZE = 100_000


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):
    if value is None:
        return ""

    return str(value).strip()


def get_tokens(value):
    """
    Input is already normalized text.
    """

    value = safe_text(value)

    if not value:
        return set()

    return {
        token
        for token in value.split()
        if len(token) >= 2
    }


def process_file(
    path,
    name_counts,
    address_counts,
):
    """
    Process one normalized Parquet source.
    """

    print("\n" + "=" * 70)
    print(f"Processing: {path.name}")
    print("=" * 70)

    parquet_file = pq.ParquetFile(path)

    total_rows = 0
    batch_number = 0

    start_time = time.time()

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=[
            "norm_name",
            "norm_address",
        ],
    ):

        batch_number += 1

        rows = batch.to_pylist()

        for row in rows:

            # ------------------------------------------------
            # Name tokens
            # ------------------------------------------------

            for token in get_tokens(
                row["norm_name"]
            ):

                name_counts[token] += 1

            # ------------------------------------------------
            # Address tokens
            # ------------------------------------------------

            for token in get_tokens(
                row["norm_address"]
            ):

                address_counts[token] += 1

        total_rows += len(rows)

        print(
            f"Batch {batch_number:>3}"
            f" | rows processed: "
            f"{total_rows:,}"
        )

        del rows

    print(
        f"\nCompleted {path.name}"
        f" in {time.time() - start_time:.1f}s"
    )

    print(
        f"Rows processed: {total_rows:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING TEST TOKEN STATISTICS")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    for path in (
        SOURCE2_PATH,
        SOURCE3_PATH,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    name_counts = Counter()
    address_counts = Counter()

    overall_start = time.time()

    # --------------------------------------------------------
    # Process test Source 2
    # --------------------------------------------------------

    process_file(
        SOURCE2_PATH,
        name_counts,
        address_counts,
    )

    # --------------------------------------------------------
    # Process test Source 3
    # --------------------------------------------------------

    process_file(
        SOURCE3_PATH,
        name_counts,
        address_counts,
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    print(
        "\nSaving token statistics..."
    )

    with open(
        NAME_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            dict(name_counts),
            file,
            ensure_ascii=False,
        )

    with open(
        ADDRESS_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            dict(address_counts),
            file,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST TOKEN STATISTICS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Unique name tokens    : "
        f"{len(name_counts):,}"
    )

    print(
        f"Unique address tokens : "
        f"{len(address_counts):,}"
    )

    print(
        "\nSaved:"
    )

    print(
        NAME_OUTPUT
    )

    print(
        ADDRESS_OUTPUT
    )

    print(
        f"\nTotal runtime: "
        f"{time.time() - overall_start:.1f}s"
    )


if __name__ == "__main__":
    main()