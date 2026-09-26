from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = PROJECT_ROOT / "output" / "normalized"


FILES = [
    "train_source1_normalized.parquet",
    "train_source2_normalized.parquet",
    "train_source3_normalized.parquet",
]


for filename in FILES:

    path = NORMALIZED_DIR / filename

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    df = pd.read_parquet(path)

    print("Rows:", f"{len(df):,}")
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(
        df[
            [
                "entity_id",
                "business_name",
                "business_address",
                "country",
                "norm_name",
                "norm_address",
                "compact_name",
            ]
        ].isna().sum()
    )

    print("\nSample:")
    print(
        df[
            [
                "entity_id",
                "business_name",
                "norm_name",
                "business_address",
                "norm_address",
                "country",
            ]
        ].head(3).to_string(index=False)
    )