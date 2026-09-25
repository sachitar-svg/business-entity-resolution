import sys
import re
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from src.preprocessing import normalize_name, normalize_address


SOURCE1_PATH = PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv"
SOURCE2_PATH = PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv"
SOURCE3_PATH = PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv"
GROUND_TRUTH_PATH = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"

CHUNK_SIZE = 200_000


# ============================================================
# THE 9 MISSED TRUE MATCHES
# ============================================================

MISSED_PAIRS = {
    "S1-867998778": {
        "S2-126598464",
        "S2-648058432",
        "S3-807085228",
    },

    "S1-42246345": {
        "S2-112471943",
        "S2-819580568",
    },

    "S1-174146241": {
        "S2-906926745",
    },

    "S1-216733182": {
        "S2-860721008",
    },

    "S1-97176033": {
        "S3-581985190",
    },

    "S1-983540069": {
        "S2-4190133",
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_tokens(value, normalizer):
    """
    Normalize text and return tokens of length >= 2.
    """
    value = normalizer(value)

    if not value:
        return set()

    return {
        token
        for token in value.split()
        if len(token) >= 2
    }


def get_numbers(value):
    """
    Extract numeric parts from an address/name.
    """
    return set(re.findall(r"\d+", str(value)))


def similarity(text1, text2):
    """
    Character-level similarity using SequenceMatcher.
    Returns a value between 0 and 100.
    """
    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(
        None,
        str(text1),
        str(text2)
    ).ratio() * 100


def compact_text(value):
    """
    Remove spaces and punctuation from normalized text.
    """
    normalized = normalize_name(value)

    return "".join(
        ch for ch in normalized
        if ch.isalnum()
    )


def print_set(values):
    """
    Print sets in a readable form.
    """
    if not values:
        return "NONE"

    return ", ".join(sorted(values))


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("=" * 80)
print("LOADING SOURCE 1")
print("=" * 80)

source1 = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype=str
).fillna("")


source1_lookup = {}

for _, row in source1.iterrows():

    s1_id = row["entity_id"]

    if s1_id in MISSED_PAIRS:
        source1_lookup[s1_id] = row.to_dict()


print(
    f"Loaded {len(source1_lookup)} of "
    f"{len(MISSED_PAIRS)} target Source 1 entities."
)


# ============================================================
# LOAD THE 9 MISSED SOURCE 2/3 RECORDS
# ============================================================

target_source_ids = set()

for ids in MISSED_PAIRS.values():
    target_source_ids.update(ids)


print()
print("=" * 80)
print("SEARCHING SOURCE 2 AND SOURCE 3")
print("=" * 80)

matched_records = {}


def scan_source(source_path, source_name):

    print(f"\nScanning {source_name}...")

    reader = pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        chunk = chunk.fillna("")

        for _, row in chunk.iterrows():

            entity_id = row["entity_id"]

            if entity_id in target_source_ids:

                matched_records[entity_id] = {
                    "source": source_name,
                    "data": row.to_dict()
                }

        if len(matched_records) == len(target_source_ids):
            break


scan_source(SOURCE2_PATH, "Source 2")

if len(matched_records) < len(target_source_ids):
    scan_source(SOURCE3_PATH, "Source 3")


print(
    f"\nFound {len(matched_records)} "
    f"of {len(target_source_ids)} missed records."
)


# ============================================================
# ANALYZE EACH MISSED PAIR
# ============================================================

print()
print("=" * 80)
print("MISSED MATCH ANALYSIS")
print("=" * 80)


overall_results = []


for s1_id, missed_ids in MISSED_PAIRS.items():

    s1 = source1_lookup[s1_id]

    s1_name = s1["business_name"]
    s1_address = s1["business_address"]
    s1_country = s1["country"]

    s1_name_norm = normalize_name(s1_name)
    s1_address_norm = normalize_address(s1_address)

    s1_name_tokens = get_tokens(
        s1_name,
        normalize_name
    )

    s1_address_tokens = get_tokens(
        s1_address,
        normalize_address
    )

    s1_numbers = get_numbers(s1_address)

    print()
    print("-" * 80)
    print(f"SOURCE 1: {s1_id}")
    print("-" * 80)

    print(f"Name    : {s1_name}")
    print(f"Address : {s1_address}")
    print(f"Country : {s1_country}")

    print(f"\nNormalized name:")
    print(s1_name_norm)

    print(f"\nNormalized address:")
    print(s1_address_norm)

    for target_id in sorted(missed_ids):

        if target_id not in matched_records:
            print()
            print(f"WARNING: {target_id} was not found.")
            continue

        record = matched_records[target_id]["data"]

        target_name = record["business_name"]
        target_address = record["business_address"]
        target_country = record["country"]

        target_name_norm = normalize_name(target_name)
        target_address_norm = normalize_address(target_address)

        target_name_tokens = get_tokens(
            target_name,
            normalize_name
        )

        target_address_tokens = get_tokens(
            target_address,
            normalize_address
        )

        target_numbers = get_numbers(target_address)

        shared_name_tokens = (
            s1_name_tokens &
            target_name_tokens
        )

        shared_address_tokens = (
            s1_address_tokens &
            target_address_tokens
        )

        shared_numbers = (
            s1_numbers &
            target_numbers
        )

        name_similarity = similarity(
            s1_name_norm,
            target_name_norm
        )

        address_similarity = similarity(
            s1_address_norm,
            target_address_norm
        )

        compact_name_similarity = similarity(
            compact_text(s1_name),
            compact_text(target_name)
        )

        same_country = (
            s1_country.strip().lower()
            ==
            target_country.strip().lower()
        )

        print()
        print(f"  TRUE MATCH: {target_id}")
        print(f"  Source    : {matched_records[target_id]['source']}")

        print(f"\n  Name:")
        print(f"    S1 : {s1_name}")
        print(f"    GT : {target_name}")

        print(f"\n  Address:")
        print(f"    S1 : {s1_address}")
        print(f"    GT : {target_address}")

        print(f"\n  Normalized name:")
        print(f"    S1 : {s1_name_norm}")
        print(f"    GT : {target_name_norm}")

        print(f"\n  Normalized address:")
        print(f"    S1 : {s1_address_norm}")
        print(f"    GT : {target_address_norm}")

        print(f"\n  Shared name tokens:")
        print(f"    {print_set(shared_name_tokens)}")

        print(f"\n  Shared address tokens:")
        print(f"    {print_set(shared_address_tokens)}")

        print(f"\n  Shared address numbers:")
        print(f"    {print_set(shared_numbers)}")

        print(f"\n  Name similarity:")
        print(f"    {name_similarity:.2f}%")

        print(f"\n  Compact-name similarity:")
        print(f"    {compact_name_similarity:.2f}%")

        print(f"\n  Address similarity:")
        print(f"    {address_similarity:.2f}%")

        print(f"\n  Same country:")
        print(f"    {same_country}")

        # ----------------------------------------------------
        # Simple pattern classification
        # ----------------------------------------------------

        patterns = []

        if shared_name_tokens:
            patterns.append("shared_name_tokens")

        if shared_address_tokens:
            patterns.append("shared_address_tokens")

        if shared_numbers:
            patterns.append("shared_numbers")

        if name_similarity >= 80:
            patterns.append("high_name_similarity")

        elif name_similarity >= 60:
            patterns.append("moderate_name_similarity")

        if address_similarity >= 80:
            patterns.append("high_address_similarity")

        elif address_similarity >= 60:
            patterns.append("moderate_address_similarity")

        if not shared_name_tokens and name_similarity < 60:
            patterns.append("strong_name_variation")

        if not shared_address_tokens and address_similarity < 60:
            patterns.append("strong_address_variation")

        print(f"\n  Observed patterns:")
        for pattern in patterns:
            print(f"    - {pattern}")

        overall_results.append({
            "s1_id": s1_id,
            "matched_id": target_id,
            "name_similarity": name_similarity,
            "address_similarity": address_similarity,
            "shared_name_tokens": len(shared_name_tokens),
            "shared_address_tokens": len(shared_address_tokens),
            "shared_numbers": len(shared_numbers),
            "same_country": same_country,
        })


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("SUMMARY OF THE 9 MISSED MATCHES")
print("=" * 80)

summary_df = pd.DataFrame(overall_results)

if not summary_df.empty:

    print(
        summary_df[
            [
                "s1_id",
                "matched_id",
                "name_similarity",
                "address_similarity",
                "shared_name_tokens",
                "shared_address_tokens",
                "shared_numbers",
            ]
        ].to_string(index=False)
    )

    print()
    print("-" * 80)

    print(
        f"Average name similarity    : "
        f"{summary_df['name_similarity'].mean():.2f}%"
    )

    print(
        f"Average address similarity : "
        f"{summary_df['address_similarity'].mean():.2f}%"
    )

    print(
        f"Matches sharing name token : "
        f"{(summary_df['shared_name_tokens'] > 0).sum()}/"
        f"{len(summary_df)}"
    )

    print(
        f"Matches sharing address token : "
        f"{(summary_df['shared_address_tokens'] > 0).sum()}/"
        f"{len(summary_df)}"
    )

    print(
        f"Matches sharing address number : "
        f"{(summary_df['shared_numbers'] > 0).sum()}/"
        f"{len(summary_df)}"
    )

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)