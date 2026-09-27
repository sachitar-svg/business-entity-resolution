from pathlib import Path
import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing import normalize_name, normalize_address


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_DIR = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "output" / "normalized"

BATCH_SIZE = 100_000


TEST_FILES = {
    "source1": (
        TEST_DIR / "test_source1.tsv",
        OUTPUT_DIR / "test_source1_normalized.parquet",
    ),
    "source2": (
        TEST_DIR / "test_source2.tsv",
        OUTPUT_DIR / "test_source2_normalized.parquet",
    ),
    "source3": (
        TEST_DIR / "test_source3.tsv",
        OUTPUT_DIR / "test_source3_normalized.parquet",
    ),
}


# ============================================================
# HELPERS
# ============================================================

def compact_text(value):
    """
    Create compact normalized business name.

    Example:
        "Prime Money"
        -> "primemoney"
    """

    normalized = normalize_name(value)

    return "".join(
        char
        for char in normalized
        if char.isalnum()
    )


def normalize_chunk(chunk):
    """
    Apply the same normalization logic used by the project.
    """

    required_columns = {
        "entity_id",
        "business_name",
        "business_address",
        "country",
    }

    missing = required_columns - set(chunk.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Keep IDs and original fields.
    result = chunk[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    ].copy()

    # Make sure entity IDs remain strings.
    result["entity_id"] = (
        result["entity_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Same normalization used by the existing pipeline.
    result["norm_name"] = (
        result["business_name"]
        .map(normalize_name)
    )

    result["norm_address"] = (
        result["business_address"]
        .map(normalize_address)
    )

    result["compact_name"] = (
        result["business_name"]
        .map(compact_text)
    )

    return result


def normalize_file(
    input_path,
    output_path,
):
    """
    Normalize one TSV file and write it as Parquet
    in batches to avoid loading the complete file.
    """

    print("\n" + "=" * 70)
    print("NORMALIZING TEST DATA")
    print("=" * 70)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove existing output so we don't append to an old file.
    if output_path.exists():
        print("\nRemoving existing output...")
        output_path.unlink()

    total_rows = 0
    batch_number = 0

    writer = None

    try:

        reader = pd.read_csv(
            input_path,
            sep="\t",
            chunksize=BATCH_SIZE,
            dtype={
                "entity_id": "string",
                "business_name": "string",
                "business_address": "string",
                "country": "string",
            },
        )

        for chunk in reader:

            batch_number += 1

            normalized = normalize_chunk(chunk)

            # Convert pandas dataframe to Arrow table.
            table = pa.Table.from_pandas(
                normalized,
                preserve_index=False,
            )

            # Create writer from the first batch.
            if writer is None:

                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)

            total_rows += len(normalized)

            print(
                f"Batch {batch_number:>3}"
                f" | rows processed: "
                f"{total_rows:,}"
            )

            del chunk
            del normalized
            del table

    finally:

        if writer is not None:
            writer.close()

    print("\nNormalization complete.")

    print(
        f"Rows written: {total_rows:,}"
    )

    print(
        f"Output: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Normalize test Source 1, Source 2, "
            "and Source 3 into Parquet files."
        )
    )

    parser.add_argument(
        "--source",
        choices=[
            "source1",
            "source2",
            "source3",
            "all",
        ],
        default="all",
        help=(
            "Which test source to normalize. "
            "Default: all."
        ),
    )

    args = parser.parse_args()

    if args.source == "all":

        selected = [
            "source1",
            "source2",
            "source3",
        ]

    else:

        selected = [
            args.source
        ]

    for source_name in selected:

        input_path, output_path = TEST_FILES[
            source_name
        ]

        normalize_file(
            input_path,
            output_path,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ALL REQUESTED TEST NORMALIZATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()