import argparse
from pathlib import Path
import pandas as pd


def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find {label} column. Available columns: {list(df.columns)}"
    )


def prepare(df):
    entity = pick_col(df, ["entity_id"], "entity_id")
    name = pick_col(
        df,
        [
            "business_name_norm",
            "normalized_business_name",
            "business_name_normalized",
            "business_name",
        ],
        "business_name",
    )
    address = pick_col(
        df,
        [
            "business_address_norm",
            "normalized_business_address",
            "business_address_normalized",
            "business_address",
        ],
        "business_address",
    )
    country = pick_col(df, ["country"], "country")

    out = df[[entity, name, address, country]].copy()
    out.columns = ["entity_id", "name_norm", "address_norm", "country"]

    for c in ["name_norm", "address_norm", "country"]:
        out[c] = out[c].fillna("").astype(str)

    out["compact_name"] = out["name_norm"].str.replace(r"\s+", "", regex=True)
    return out


def build_index(src):
    src = src.drop_duplicates("entity_id").reset_index(drop=True)

    print("Calculating key frequencies...")

    name_freq = src.loc[src.name_norm.ne(""), "name_norm"].value_counts()
    addr_freq = src.loc[src.address_norm.ne(""), "address_norm"].value_counts()
    compact_freq = src.loc[src.compact_name.ne(""), "compact_name"].value_counts()

    # Keep only reasonably selective exact keys.
    # Empty strings are excluded.
    name_ok = src.name_norm.ne("") & src.name_norm.map(name_freq).le(20)
    addr_ok = src.address_norm.ne("") & src.address_norm.map(addr_freq).le(20)
    compact_ok = (
        src.compact_name.str.len().ge(8)
        & src.compact_name.map(compact_freq).le(20)
    )

    name_idx = src.loc[
        name_ok, ["name_norm", "country", "entity_id"]
    ].copy()
    name_idx["blocker"] = "name"

    addr_idx = src.loc[
        addr_ok, ["address_norm", "country", "entity_id"]
    ].copy()
    addr_idx["blocker"] = "address"

    compact_idx = src.loc[
        compact_ok, ["compact_name", "country", "entity_id"]
    ].copy()
    compact_idx["blocker"] = "compact"

    return name_idx, addr_idx, compact_idx


def process_chunk(s1, name_idx, addr_idx, compact_idx, top_k):
    s1 = s1.copy()
    s1["_row"] = range(len(s1))

    pieces = []

    # Exact normalized name.
    q = s1.loc[
        s1.name_norm.ne(""),
        ["_row", "name_norm", "country"],
    ].merge(
        name_idx,
        on=["name_norm", "country"],
        how="inner",
    )
    if not q.empty:
        pieces.append(q[["_row", "entity_id", "blocker"]])

    # Exact normalized address.
    q = s1.loc[
        s1.address_norm.ne(""),
        ["_row", "address_norm", "country"],
    ].merge(
        addr_idx,
        on=["address_norm", "country"],
        how="inner",
    )
    if not q.empty:
        pieces.append(q[["_row", "entity_id", "blocker"]])

    # Compact name: catches spacing differences, but only for sufficiently
    # long and selective keys.
    q = s1.loc[
        s1.compact_name.str.len().ge(8),
        ["_row", "compact_name", "country"],
    ].merge(
        compact_idx,
        on=["compact_name", "country"],
        how="inner",
    )
    if not q.empty:
        pieces.append(q[["_row", "entity_id", "blocker"]])

    if pieces:
        raw = pd.concat(pieces, ignore_index=True)
        raw = raw.drop_duplicates(["_row", "entity_id", "blocker"])

        # Candidate ranking: exact name first, then address, then compact.
        priority_map = {"name": 1, "address": 2, "compact": 3}
        raw["priority"] = raw["blocker"].map(priority_map)

        cand = (
            raw.groupby(["_row", "entity_id"], as_index=False)
            .agg(
                priority=("priority", "min"),
                blockers=("blocker", lambda x: ",".join(sorted(set(x)))),
            )
        )

        cand = cand.sort_values(
            ["_row", "priority", "entity_id"],
            kind="stable",
        )
        cand = cand.groupby("_row", sort=False).head(top_k)

        # For deterministic matching, use conservative rules:
        # 1) candidate supported by BOTH exact name and exact address
        # 2) unique exact-name candidate
        # 3) unique exact-address candidate
        # 4) unique compact-name candidate
        matched_rows = {}

        for row_id, g in cand.groupby("_row", sort=False):
            blockers = g["blockers"].tolist()

            both = g.loc[
                g["blockers"].str.contains("name", regex=False)
                & g["blockers"].str.contains("address", regex=False),
                "entity_id",
            ].tolist()

            if both:
                matched_rows[row_id] = ",".join(map(str, both))
                continue

            name_only = g.loc[g["blockers"].eq("name"), "entity_id"].tolist()
            if len(name_only) == 1:
                matched_rows[row_id] = str(name_only[0])
                continue

            addr_only = g.loc[g["blockers"].eq("address"), "entity_id"].tolist()
            if len(addr_only) == 1:
                matched_rows[row_id] = str(addr_only[0])
                continue

            compact_only = g.loc[
                g["blockers"].eq("compact"), "entity_id"
            ].tolist()
            if len(g) == 1 and len(compact_only) == 1:
                matched_rows[row_id] = str(compact_only[0])

        candidate_lists = (
            cand.groupby("_row", sort=False)["entity_id"]
            .agg(lambda x: ",".join(map(str, x)))
            .to_dict()
        )
    else:
        cand = pd.DataFrame()
        candidate_lists = {}
        matched_rows = {}

    candidate_values = [
        candidate_lists.get(i, "") for i in range(len(s1))
    ]
    match_values = [
        matched_rows.get(i, "") for i in range(len(s1))
    ]

    cand_out = pd.DataFrame({
        "source1_entity_id": s1["entity_id"].astype(str),
        "candidate_entity_ids": candidate_values,
    })

    match_out = pd.DataFrame({
        "source1_entity_id": s1["entity_id"].astype(str),
        "matched_entity_ids": match_values,
    })

    return cand_out, match_out


def main():
    parser = argparse.ArgumentParser(
        description="Fast emergency submission generator."
    )
    parser.add_argument(
        "--s1",
        default="output/normalized/test_source1_normalized.parquet",
    )
    parser.add_argument(
        "--s2",
        default="output/normalized/test_source2_normalized.parquet",
    )
    parser.add_argument(
        "--s3",
        default="output/normalized/test_source3_normalized.parquet",
    )
    parser.add_argument(
        "--candidate-output",
        default="output/candidate_pairs.tsv",
    )
    parser.add_argument(
        "--match-output",
        default="output/matching_results.tsv",
    )
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    print("Loading Source2...")
    s2_raw = pd.read_parquet(args.s2)
    s2 = prepare(s2_raw)
    del s2_raw
    print(f"Source2 rows: {len(s2):,}")

    print("Loading Source3...")
    s3_raw = pd.read_parquet(args.s3)
    s3 = prepare(s3_raw)
    del s3_raw
    print(f"Source3 rows: {len(s3):,}")

    src = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    print(f"Combined Source2+3 rows: {len(src):,}")
    print("Building lightweight indexes...")
    name_idx, addr_idx, compact_idx = build_index(src)
    del src

    print(f"Name index rows: {len(name_idx):,}")
    print(f"Address index rows: {len(addr_idx):,}")
    print(f"Compact-name index rows: {len(compact_idx):,}")

    candidate_path = Path(args.candidate_output)
    match_path = Path(args.match_output)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    match_path.parent.mkdir(parents=True, exist_ok=True)

    # Read Source1 in batches so the full 1.73M-row table is not held in RAM.
    print("\nOpening Source1 in batches...")
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(args.s1)
    total_rows = parquet_file.metadata.num_rows

    first = True
    total_match_rows = 0
    total_candidate_rows = 0
    processed = 0
    for batch in parquet_file.iter_batches(batch_size=args.chunk_size):
        raw = batch.to_pandas()
        s1 = prepare(raw)
        del raw

        cand_out, match_out = process_chunk(
            s1,
            name_idx,
            addr_idx,
            compact_idx,
            args.top_k,
        )

        cand_out.to_csv(
            candidate_path,
            sep="\t",
            index=False,
            mode="w" if first else "a",
            header=first,
        )
        match_out.to_csv(
            match_path,
            sep="\t",
            index=False,
            mode="w" if first else "a",
            header=first,
        )

        first = False
        processed += len(s1)
        total_candidate_rows += int(
            cand_out["candidate_entity_ids"].ne("").sum()
        )
        total_match_rows += int(
            match_out["matched_entity_ids"].ne("").sum()
        )

        print(
            f"Processed S1: {processed:,} / {total_rows:,} "
            f"| candidate rows: {total_candidate_rows:,} "
            f"| matched rows: {total_match_rows:,}"
        )

        del s1, cand_out, match_out

    print("\nFAST SUBMISSION COMPLETE")
    print(f"Candidate file: {candidate_path}")
    print(f"Match file     : {match_path}")
    print(f"Rows written   : {processed:,}")


if __name__ == "__main__":
    main()
