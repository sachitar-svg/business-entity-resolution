import pandas as pd
from pathlib import Path

BASE = Path(".")

S1_FILE = BASE / "output/normalized/test_source1_normalized.parquet"

MATCH_IN = BASE / "output/matching_results.tsv"
MATCH_OUT = BASE / "output/matching_results_complete.tsv"

CAND_IN = BASE / "output/candidate_pairs.tsv"
CAND_OUT = BASE / "output/candidate_pairs_complete.tsv"


def complete_file(s1_ids, input_file, output_file, id_col, value_col):
    print(f"\nProcessing: {input_file}")

    if not input_file.exists():
        print(f"WARNING: {input_file} does not exist. Skipping.")
        return

    df = pd.read_csv(
        input_file,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    required = {id_col, value_col}
    missing_cols = required - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"{input_file} is missing columns: {missing_cols}. "
            f"Found: {list(df.columns)}"
        )

    # Remove duplicate S1 IDs from the partial output.
    df = df.drop_duplicates(subset=[id_col], keep="last")

    # Keep exactly the official Source1 ID list and its original order.
    official = pd.DataFrame({id_col: s1_ids})

    complete = official.merge(
        df[[id_col, value_col]],
        on=id_col,
        how="left"
    )

    complete[value_col] = complete[value_col].fillna("")

    complete.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    missing = (complete[value_col] == "").sum()

    print(f"Written: {output_file}")
    print(f"Rows: {len(complete):,}")
    print(f"Blank {value_col}: {missing:,}")


def main():
    print("Loading official test Source1 IDs...")

    s1 = pd.read_parquet(
        S1_FILE,
        columns=["entity_id"]
    )

    s1_ids = s1["entity_id"].astype(str).tolist()

    print(f"Official Source1 rows: {len(s1_ids):,}")

    complete_file(
        s1_ids,
        MATCH_IN,
        MATCH_OUT,
        "source1_entity_id",
        "matched_entity_ids"
    )

    complete_file(
        s1_ids,
        CAND_IN,
        CAND_OUT,
        "source1_entity_id",
        "candidate_entity_ids"
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()