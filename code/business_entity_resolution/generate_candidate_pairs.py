import sys
from pathlib import Path
import pandas as pd
import re


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "dataset" / "train"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 1000
RANDOM_STATE = 42
CHUNK_SIZE = 100_000


# ============================================================
# IMPORT PREPROCESSING
# ============================================================

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from src.preprocessing import normalize_name, normalize_address


# ============================================================
# LOAD SOURCE 1 + GROUND TRUTH
# ============================================================

print("Loading Source 1 and ground truth...")

source1 = pd.read_parquet(
    DATA_DIR / "train_source1.parquet"
)

ground_truth = pd.read_parquet(
    DATA_DIR / "train_ground_truth.parquet"
)

print(f"Source 1 rows: {len(source1):,}")
print(f"Ground truth rows: {len(ground_truth):,}")


# ============================================================
# SAMPLE SOURCE 1
# ============================================================

sample_s1 = source1.sample(
    n=min(SAMPLE_SIZE, len(source1)),
    random_state=RANDOM_STATE
).copy()

print(f"\nUsing {len(sample_s1):,} Source 1 entities")


# ============================================================
# BUILD GROUND-TRUTH LOOKUP
# ============================================================

gt_lookup = {}

for row in ground_truth.itertuples(index=False):

    value = row.matched_entity_ids

    if pd.isna(value) or str(value).strip() == "":
        matches = set()
    else:
        matches = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    gt_lookup[row.source1_entity_id] = matches


# ============================================================
# PREPARE SOURCE 1
# ============================================================

sample_s1["norm_name"] = (
    sample_s1["business_name"]
    .fillna("")
    .map(normalize_name)
)

sample_s1["norm_address"] = (
    sample_s1["business_address"]
    .fillna("")
    .map(normalize_address)
)

sample_s1["compact_name"] = (
    sample_s1["norm_name"]
    .str.replace(" ", "", regex=False)
)

sample_s1["name_tokens"] = sample_s1["norm_name"].map(
    lambda x: set(x.split()) if x else set()
)

sample_s1["address_tokens"] = sample_s1["norm_address"].map(
    lambda x: set(x.split()) if x else set()
)


# ============================================================
# BUILD SIMPLE BLOCKING INDEXES
# ============================================================

print("\nBuilding blocking indexes...")

name_index = {}
address_index = {}
country_index = {}

for row in sample_s1.itertuples(index=False):

    entity_id = row.entity_id

    # Exact normalized name
    if row.norm_name:
        name_index.setdefault(row.norm_name, set()).add(entity_id)

    # Address tokens
    for token in row.address_tokens:
        if token:
            address_index.setdefault(token, set()).add(entity_id)

    # Country
    if row.country:
        country_index.setdefault(row.country, set()).add(entity_id)


print(f"Name index keys: {len(name_index):,}")
print(f"Address index keys: {len(address_index):,}")
print(f"Country index keys: {len(country_index):,}")


# ============================================================
# SCAN SOURCE 2 AND SOURCE 3
# ============================================================

candidate_pairs = []

sample_ids = set(sample_s1["entity_id"])


def process_source(source_path, source_name):

    print(f"\nScanning {source_name}...")

    reader = pd.read_parquet(
        source_path
    )

    total_candidates = 0

    for row in reader.itertuples(index=False):

        entity_id = row.entity_id
        name = normalize_name(row.business_name)
        address = normalize_address(row.business_address)
        country = row.country

        candidates = set()

        # ----------------------------------------------------
        # Exact normalized name
        # ----------------------------------------------------

        if name:
            candidates.update(
                name_index.get(name, set())
            )

        # ----------------------------------------------------
        # Address token blocking
        # ----------------------------------------------------

        address_tokens = set(address.split())

        for token in address_tokens:
            candidates.update(
                address_index.get(token, set())
            )

        # ----------------------------------------------------
        # Country filter
        # ----------------------------------------------------

        if country:
            country_candidates = country_index.get(
                country,
                set()
            )

            if candidates:
                candidates &= country_candidates
            else:
                candidates = set()

        # ----------------------------------------------------
        # Save candidate pairs
        # ----------------------------------------------------

        for s1_id in candidates:

            candidate_pairs.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": entity_id,
                "candidate_source": source_name,
                "label": int(
                    entity_id in gt_lookup.get(s1_id, set())
                )
            })

        total_candidates += len(candidates)

    print(
        f"  Candidates generated: {total_candidates:,}"
    )


process_source(
    DATA_DIR / "train_source2.parquet",
    "source2"
)

process_source(
    DATA_DIR / "train_source3.parquet",
    "source3"
)


# ============================================================
# SAVE RESULTS
# ============================================================

candidate_df = pd.DataFrame(candidate_pairs)

output_path = OUTPUT_DIR / "candidate_pairs_sample.parquet"

candidate_df.to_parquet(
    output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("CANDIDATE GENERATION COMPLETE")
print("=" * 50)

print(f"Candidate pairs: {len(candidate_df):,}")
print(
    f"Positive matches: "
    f"{candidate_df['label'].sum():,}"
)

print(
    f"Negative candidates: "
    f"{(candidate_df['label'] == 0).sum():,}"
)

print(f"\nSaved to:")
print(output_path)