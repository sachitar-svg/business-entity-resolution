from pathlib import Path
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
OUTPUT_DIR = PROJECT_ROOT / "output" / "normalized"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMPORT PREPROCESSING
# ============================================================

SRC_DIR = PROJECT_ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from src.preprocessing import normalize_name, normalize_address


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 100_000


# ============================================================
# FILES
# ============================================================

SOURCE_FILES = {
    "train_source1": DATA_DIR / "train_source1.parquet",
    "train_source2": DATA_DIR / "train_source2.parquet",
    "train_source3": DATA_DIR / "train_source3.parquet",
}


# ============================================================
# NORMALIZE ONE PARQUET FILE
# ============================================================

def normalize_parquet(input_path, output_path):

    print("\n" + "=" * 60)
    print(f"Processing: {input_path.name}")
    print("=" * 60)

    if output_path.exists():
        print(f"Already exists: {output_path}")
        print("Skipping.")
        return

    start_time = time.time()

    parquet_file = pq.ParquetFile(input_path)

    total_rows = 0
    batch_number = 0

    writer = None

    try:
        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):

            batch_number += 1

            # Convert only the current batch to pandas
            df = batch.to_pandas()

            # Make sure missing values are handled
            df["business_name"] = df["business_name"].fillna("")
            df["business_address"] = df["business_address"].fillna("")
            df["country"] = df["country"].fillna("")

            # ------------------------------------------------
            # NORMALIZATION
            # ------------------------------------------------

            df["norm_name"] = df["business_name"].map(
                normalize_name
            )

            df["norm_address"] = df["business_address"].map(
                normalize_address
            )

            df["compact_name"] = (
                df["norm_name"]
                .str.replace(" ", "", regex=False)
            )

            # ------------------------------------------------
            # WRITE CURRENT BATCH
            # ------------------------------------------------

            table = pa.Table.from_pandas(
                df,
                preserve_index=False
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy"
                )

            writer.write_table(table)

            total_rows += len(df)

            print(
                f"  Batch {batch_number:>3}: "
                f"{total_rows:,} rows processed"
            )

            # Release batch memory
            del df
            del table

    finally:

        if writer is not None:
            writer.close()

    elapsed = time.time() - start_time

    print("\nCompleted:")
    print(f"  Rows      : {total_rows:,}")
    print(f"  Time      : {elapsed:.1f} seconds")
    print(f"  Output    : {output_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Starting normalized Parquet generation...")
    print(f"Batch size: {BATCH_SIZE:,}")

    overall_start = time.time()

    for name, input_path in SOURCE_FILES.items():

        output_path = OUTPUT_DIR / f"{name}_normalized.parquet"

        normalize_parquet(
            input_path=input_path,
            output_path=output_path
        )

    print("\n" + "=" * 60)
    print("ALL NORMALIZED DATASETS READY")
    print("=" * 60)

    print(
        f"Total time: "
        f"{time.time() - overall_start:.1f} seconds"
    )

    print("\nFiles created in:")
    print(OUTPUT_DIR)