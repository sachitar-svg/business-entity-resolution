from pathlib import Path
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_TEST_DIR = (
    PROJECT_ROOT.parent
    / "business-entity-resolution"
    / "dataset"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "normalized_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SRC_DIR = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "src"
)

sys.path.insert(0, str(SRC_DIR))

from preprocessing import normalize_name, normalize_address


BATCH_SIZE = 100_000


SOURCE_FILES = {
    "test_source1": RAW_TEST_DIR / "test_source1.tsv",
    "test_source2": RAW_TEST_DIR / "test_source2.tsv",
    "test_source3": RAW_TEST_DIR / "test_source3.tsv",
}


def normalize_tsv(input_path: Path, output_path: Path):

    print("\n" + "=" * 70)
    print(f"Processing: {input_path}")
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}"
        )

    if output_path.exists():
        print(f"Already exists: {output_path}")
        print("Skipping.")
        return

    start_time = time.time()

    total_rows = 0
    batch_number = 0
    writer = None

    try:
        reader = pd.read_csv(
            input_path,
            sep="\t",
            dtype=str,
            encoding="utf-8",
            chunksize=BATCH_SIZE,
            keep_default_na=False,
        )

        for df in reader:

            batch_number += 1

            required_columns = {
                "entity_id",
                "business_name",
                "business_address",
                "country",
            }

            missing_columns = (
                required_columns - set(df.columns)
            )

            if missing_columns:
                raise ValueError(
                    f"Missing required columns in "
                    f"{input_path.name}: "
                    f"{sorted(missing_columns)}"
                )

            for column in [
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ]:
                df[column] = (
                    df[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

            df["norm_name"] = (
                df["business_name"]
                .map(normalize_name)
            )

            df["norm_address"] = (
                df["business_address"]
                .map(normalize_address)
            )

            df["compact_name"] = (
                df["norm_name"]
                .str.replace(" ", "", regex=False)
            )

            table = pa.Table.from_pandas(
                df,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)

            total_rows += len(df)

            print(
                f"  Batch {batch_number:>3}"
                f" | rows processed: "
                f"{total_rows:,}"
            )

            del df
            del table

    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - start_time

    print("\nCompleted:")
    print(f"  Rows   : {total_rows:,}")
    print(f"  Time   : {elapsed:.1f} seconds")
    print(f"  Output : {output_path}")


if __name__ == "__main__":

    print("=" * 70)
    print("TEST NORMALIZED PARQUET GENERATION")
    print("=" * 70)

    print(f"Raw test directory : {RAW_TEST_DIR}")
    print(f"Output directory   : {OUTPUT_DIR}")
    print(f"Batch size         : {BATCH_SIZE:,}")

    overall_start = time.time()

    for name, input_path in SOURCE_FILES.items():

        output_path = (
            OUTPUT_DIR
            / f"{name}_normalized.parquet"
        )

        normalize_tsv(
            input_path,
            output_path,
        )

    print("\n" + "=" * 70)
    print("ALL TEST NORMALIZED DATASETS READY")
    print("=" * 70)

    print(
        f"Total time: "
        f"{time.time() - overall_start:.1f} seconds"
    )

    print("\nFiles created in:")
    print(OUTPUT_DIR)
