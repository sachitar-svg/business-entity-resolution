from pathlib import Path
import argparse
import difflib
import re

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz


PROJECT_ROOT = Path(__file__).resolve().parents[3]

S1_DEFAULT = PROJECT_ROOT / "output" / "normalized" / "test_source1_normalized.parquet"
S2_DEFAULT = PROJECT_ROOT / "output" / "normalized" / "test_source2_normalized.parquet"
S3_DEFAULT = PROJECT_ROOT / "output" / "normalized" / "test_source3_normalized.parquet"
CANDIDATE_DEFAULT = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
MODEL_DEFAULT = PROJECT_ROOT / "output" / "tree_matching_model.joblib"
OUTPUT_DEFAULT = PROJECT_ROOT / "output" / "matching_results.tsv"

FEATURE_COLUMNS = [
    "name_exact",
    "name_compact_exact",
    "name_token_jaccard",
    "name_char_similarity",
    "address_token_jaccard",
    "address_char_similarity",
    "number_overlap",
    "number_jaccard",
    "country_match",
    "s1_name_missing",
    "candidate_name_missing",
    "s1_address_missing",
    "candidate_address_missing",
]


def safe_text(value):
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def compact_from_norm(value):
    return "".join(ch for ch in safe_text(value) if ch.isalnum())


def token_set(value):
    value = safe_text(value)
    if not value:
        return frozenset()
    return frozenset(tok for tok in value.split() if len(tok) >= 2)


def number_set(value):
    value = safe_text(value)
    if not value:
        return frozenset()

    out = set()
    for n in re.findall(r"\d+", value):
        n = n.lstrip("0")
        if n == "":
            n = "0"
        out.add(n)
    return frozenset(out)


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sequence_ratio(a, b):
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def read_normalized(path):
    path = Path(path)
    names = set(pq.ParquetFile(path).schema_arrow.names)

    if "entity_id" not in names:
        raise ValueError(f"{path} has no entity_id column.")

    name_col = next(
        (c for c in ["norm_name", "business_name_norm",
                     "normalized_business_name", "business_name"]
         if c in names),
        None,
    )
    addr_col = next(
        (c for c in ["norm_address", "business_address_norm",
                     "normalized_business_address", "business_address"]
         if c in names),
        None,
    )

    if name_col is None or addr_col is None or "country" not in names:
        raise ValueError(
            f"{path} does not contain the expected name/address/country columns. "
            f"Found: {sorted(names)}"
        )

    wanted = ["entity_id", name_col, addr_col, "country"]
    if "compact_name" in names:
        wanted.append("compact_name")

    df = pd.read_parquet(path, columns=wanted)

    rename = {
        name_col: "name",
        addr_col: "address",
    }
    df = df.rename(columns=rename)

    if "compact_name" not in df.columns:
        df["compact_name"] = df["name"].map(compact_from_norm)
    else:
        df["compact_name"] = df["compact_name"].fillna("").astype(str)

    for c in ["entity_id", "name", "address", "country", "compact_name"]:
        df[c] = df[c].fillna("").astype(str)

    df = df.drop_duplicates(subset=["entity_id"], keep="first")
    return df


def prepare_arrays(pair_df):
    s_name = pair_df["s_name"].to_numpy(dtype=object)
    c_name = pair_df["c_name"].to_numpy(dtype=object)
    s_compact = pair_df["s_compact"].to_numpy(dtype=object)
    c_compact = pair_df["c_compact"].to_numpy(dtype=object)
    s_addr = pair_df["s_address"].to_numpy(dtype=object)
    c_addr = pair_df["c_address"].to_numpy(dtype=object)
    s_country = pair_df["s_country"].to_numpy(dtype=object)
    c_country = pair_df["c_country"].to_numpy(dtype=object)

    n = len(pair_df)

    name_exact = np.empty(n, dtype=np.float32)
    name_compact_exact = np.empty(n, dtype=np.float32)
    name_token_jaccard = np.empty(n, dtype=np.float32)
    name_char_similarity = np.empty(n, dtype=np.float32)
    address_token_jaccard = np.empty(n, dtype=np.float32)
    address_char_similarity = np.empty(n, dtype=np.float32)
    number_overlap = np.empty(n, dtype=np.float32)
    number_jaccard = np.empty(n, dtype=np.float32)
    country_match = np.empty(n, dtype=np.float32)

    # Cache prepared candidate-side strings/features within this chunk.
    cand_cache = {}

    for i in range(n):
        sn = safe_text(s_name[i])
        cn = safe_text(c_name[i])
        sc = safe_text(s_compact[i])
        cc = safe_text(c_compact[i])
        sa = safe_text(s_addr[i])
        ca = safe_text(c_addr[i])

        # Exact and compact exact.
        name_exact[i] = float(bool(sn and cn and sn == cn))
        name_compact_exact[i] = float(bool(sc and cc and sc == cc))

        # Token Jaccard.
        nt_s = token_set(sn)
        nt_c = token_set(cn)
        name_token_jaccard[i] = jaccard(nt_s, nt_c)

        at_s = token_set(sa)
        at_c = token_set(ca)
        address_token_jaccard[i] = jaccard(at_s, at_c)

        # Character similarity:
        # Use RapidFuzz's C++ implementation for speed. It is a very close
        # character-similarity proxy to the training-time SequenceMatcher
        # feature, and is substantially faster over millions of pairs.
        name_char_similarity[i] = (
            fuzz.ratio(sn, cn) / 100.0 if sn and cn else 0.0
        )
        address_char_similarity[i] = (
            fuzz.ratio(sa, ca) / 100.0 if sa and ca else 0.0
        )

        ns = number_set(sa)
        nc = number_set(ca)
        inter = ns & nc
        union = ns | nc
        number_overlap[i] = float(bool(inter))
        number_jaccard[i] = (
            len(inter) / len(union) if union else 0.0
        )

        country_match[i] = float(
            bool(
                safe_text(s_country[i])
                and safe_text(c_country[i])
                and safe_text(s_country[i]).casefold()
                == safe_text(c_country[i]).casefold()
            )
        )

    features = np.column_stack([
        name_exact,
        name_compact_exact,
        name_token_jaccard,
        name_char_similarity,
        address_token_jaccard,
        address_char_similarity,
        number_overlap,
        number_jaccard,
        country_match,
        (s_name == "").astype(np.float32),
        (c_name == "").astype(np.float32),
        (s_addr == "").astype(np.float32),
        (c_addr == "").astype(np.float32),
    ])

    return features.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Score the already-generated candidate_pairs.tsv with Tree V3."
    )
    parser.add_argument("--s1", default=str(S1_DEFAULT))
    parser.add_argument("--s2", default=str(S2_DEFAULT))
    parser.add_argument("--s3", default=str(S3_DEFAULT))
    parser.add_argument("--candidates", default=str(CANDIDATE_DEFAULT))
    parser.add_argument("--model", default=str(MODEL_DEFAULT))
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--threshold", type=float, default=0.951)
    parser.add_argument("--candidate-chunk", type=int, default=20000)
    parser.add_argument(
        "--char-sim",
        choices=["rapidfuzz", "difflib"],
        default="rapidfuzz",
        help="rapidfuzz is much faster; difflib matches training exactly but is slower.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TREE V3 SCORING OF EXISTING CANDIDATES")
    print("=" * 70)
    print(f"Candidates : {args.candidates}")
    print(f"Threshold  : {args.threshold}")
    print(f"Chunk S1   : {args.candidate_chunk:,}")
    print()

    print("Loading Tree V3...")
    bundle = joblib.load(args.model)
    model = bundle["model"]
    model_features = bundle["feature_columns"]

    if list(model_features) != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature columns do not match expected Tree V3 feature order.\n"
            f"Model: {model_features}\n"
            f"Expected: {FEATURE_COLUMNS}"
        )
    print("Tree V3 feature order: PASS")

    print("\nLoading Source1...")
    s1 = read_normalized(args.s1)
    print(f"Source1 rows: {len(s1):,}")

    print("Loading Source2...")
    s2 = read_normalized(args.s2)
    print(f"Source2 rows: {len(s2):,}")

    print("Loading Source3...")
    s3 = read_normalized(args.s3)
    print(f"Source3 rows: {len(s3):,}")

    source = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    source = source.set_index("entity_id", drop=False)
    s1 = s1.set_index("entity_id", drop=False)

    print(f"Combined candidate-side rows: {len(source):,}")

    # Accumulate accepted matches by Source1 ID.
    accepted = {}
    processed_rows = 0
    scored_pairs = 0
    accepted_pairs = 0

    candidate_reader = pd.read_csv(
        args.candidates,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=args.candidate_chunk,
    )

    for block_no, cand in enumerate(candidate_reader, start=1):
        processed_rows += len(cand)

        cand = cand[cand["candidate_entity_ids"].ne("")].copy()

        if cand.empty:
            print(
                f"Block {block_no}: S1 processed {processed_rows:,} "
                f"| no candidate rows"
            )
            continue

        # Turn comma-separated candidates into individual pairs.
        cand["candidate_entity_ids"] = cand["candidate_entity_ids"].str.split(",")
        pairs = cand[[
            "source1_entity_id",
            "candidate_entity_ids",
        ]].explode(
            "candidate_entity_ids",
            ignore_index=True,
        )

        pairs = pairs.rename(
            columns={"candidate_entity_ids": "candidate_id"}
        )
        pairs["source1_entity_id"] = pairs["source1_entity_id"].astype(str)
        pairs["candidate_id"] = pairs["candidate_id"].astype(str)

        # Drop duplicate pair IDs in case the candidate list contains one.
        pairs = pairs.drop_duplicates(
            subset=["source1_entity_id", "candidate_id"],
            keep="first",
        )

        scored_pairs += len(pairs)

        # Lookup the two records.
        s_part = s1.reindex(pairs["source1_entity_id"].tolist())
        c_part = source.reindex(pairs["candidate_id"].tolist())

        pair_df = pd.DataFrame({
            "source1_entity_id": pairs["source1_entity_id"].to_numpy(),
            "candidate_id": pairs["candidate_id"].to_numpy(),

            "s_name": s_part["name"].to_numpy(),
            "s_compact": s_part["compact_name"].to_numpy(),
            "s_address": s_part["address"].to_numpy(),
            "s_country": s_part["country"].to_numpy(),

            "c_name": c_part["name"].to_numpy(),
            "c_compact": c_part["compact_name"].to_numpy(),
            "c_address": c_part["address"].to_numpy(),
            "c_country": c_part["country"].to_numpy(),
        })

        # Unknown candidate IDs are not expected, but skip them safely.
        valid = pair_df["c_name"].notna() & pair_df["c_address"].notna()
        pair_df = pair_df.loc[valid].reset_index(drop=True)

        if pair_df.empty:
            continue

        X = prepare_arrays(pair_df)

        probabilities = model.predict_proba(X)[:, 1]
        keep = probabilities >= args.threshold

        kept = pair_df.loc[
            keep,
            ["source1_entity_id", "candidate_id"],
        ].copy()

        accepted_pairs += len(kept)

        for sid, group in kept.groupby("source1_entity_id", sort=False):
            ids = group["candidate_id"].astype(str).tolist()
            previous = accepted.get(str(sid), [])
            previous.extend(ids)
            accepted[str(sid)] = previous

        if block_no == 1 or block_no % 5 == 0:
            print(
                f"Block {block_no}: S1 processed {processed_rows:,} "
                f"| pairs scored {scored_pairs:,} "
                f"| accepted pairs {accepted_pairs:,}"
            )

    # Deduplicate accepted IDs while preserving sorted deterministic output.
    for sid, ids in accepted.items():
        accepted[sid] = sorted(set(ids))

    # Write exactly one row for every official Source1 entity.
    result = pd.DataFrame({
        "source1_entity_id": s1["entity_id"].astype(str).tolist()
    })
    result["matched_entity_ids"] = result["source1_entity_id"].map(
        lambda sid: ",".join(accepted.get(sid, []))
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 70)
    print("TREE V3 SCORING COMPLETE")
    print("=" * 70)
    print(f"Output              : {output_path}")
    print(f"Output rows         : {len(result):,}")
    print(f"Unique S1           : {result['source1_entity_id'].nunique():,}")
    print(
        f"Matched S1          : "
        f"{int(result['matched_entity_ids'].ne('').sum()):,}"
    )
    print(f"Pairs scored        : {scored_pairs:,}")
    print(f"Accepted pairs      : {accepted_pairs:,}")
    print("Candidate file      : unchanged")


if __name__ == "__main__":
    main()
