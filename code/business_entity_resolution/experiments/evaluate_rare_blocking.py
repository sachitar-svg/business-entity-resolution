import json
from collections import defaultdict

import pandas as pd

from preprocessing import normalize_name, normalize_address


# --------------------------------------------------
# Configuration
# --------------------------------------------------

THRESHOLDS = [10, 25, 50, 100, 250, 500]

TOP_K_NAME = 2
TOP_K_ADDRESS = 2

CHUNK_SIZE = 200_000

NAME_STATS_PATH = (
    "code/business_entity_resolution/data/token_stats/"
    "name_token_counts.json"
)

ADDRESS_STATS_PATH = (
    "code/business_entity_resolution/data/token_stats/"
    "address_token_counts.json"
)


# --------------------------------------------------
# Load token statistics
# --------------------------------------------------

with open(NAME_STATS_PATH, "r", encoding="utf-8") as f:
    name_counts = json.load(f)

with open(ADDRESS_STATS_PATH, "r", encoding="utf-8") as f:
    address_counts = json.load(f)


# --------------------------------------------------
# Load sample S1 entities
# --------------------------------------------------

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=5
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_tokens(value, normalizer):
    """Return unique normalized tokens."""

    normalized = normalizer(value)

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= 2
    }


def choose_rare_tokens(tokens, counts, max_frequency, top_k):
    """
    Choose the rarest tokens that actually occur in S2/S3
    and whose frequency is <= max_frequency.
    """

    eligible = []

    for token in tokens:

        frequency = counts.get(token)

        # Token does not occur in S2/S3.
        if frequency is None:
            continue

        if frequency <= max_frequency:
            eligible.append(
                (token, frequency)
            )

    # Rarest first
    eligible.sort(
        key=lambda x: x[1]
    )

    return [
        token
        for token, _ in eligible[:top_k]
    ]


# --------------------------------------------------
# Build lookup:
#
# token -> {(S1 ID, threshold)}
#
# This allows one scan of S2/S3 to evaluate
# multiple thresholds.
# --------------------------------------------------

name_lookup = defaultdict(set)
address_lookup = defaultdict(set)

query_info = {}


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    country = str(
        row["country"]
    ).casefold().strip()

    name_tokens = get_tokens(
        row["business_name"],
        normalize_name
    )

    address_tokens = get_tokens(
        row["business_address"],
        normalize_address
    )

    query_info[s1_id] = {
        "country": country,
        "name_keys": {},
        "address_keys": {}
    }

    for threshold in THRESHOLDS:

        name_keys = choose_rare_tokens(
            name_tokens,
            name_counts,
            threshold,
            TOP_K_NAME
        )

        address_keys = choose_rare_tokens(
            address_tokens,
            address_counts,
            threshold,
            TOP_K_ADDRESS
        )

        query_info[s1_id]["name_keys"][threshold] = name_keys
        query_info[s1_id]["address_keys"][threshold] = address_keys

        for token in name_keys:
            name_lookup[token].add(
                (s1_id, threshold)
            )

        for token in address_keys:
            address_lookup[token].add(
                (s1_id, threshold)
            )


# --------------------------------------------------
# Print selected blocking keys
# --------------------------------------------------

print("\n===== SELECTED BLOCKING KEYS =====")

for s1_id, info in query_info.items():

    print("\nS1:", s1_id)

    for threshold in THRESHOLDS:

        print(
            f"  threshold={threshold:>3} | "
            f"name={info['name_keys'][threshold]} | "
            f"address={info['address_keys'][threshold]}"
        )


# --------------------------------------------------
# Candidate storage
# --------------------------------------------------

candidate_sets = defaultdict(set)


# --------------------------------------------------
# Scan S2 and S3 once
# --------------------------------------------------

source_paths = [
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv"
]


for source_path in source_paths:

    print(f"\nProcessing: {source_path}")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            source_path,
            sep="\t",
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"  Chunk {chunk_number}"
        )

        for _, row in chunk.iterrows():

            entity_id = row["entity_id"]

            country = str(
                row["country"]
            ).casefold().strip()

            # Normalize the candidate once
            candidate_name_tokens = get_tokens(
                row["business_name"],
                normalize_name
            )

            candidate_address_tokens = get_tokens(
                row["business_address"],
                normalize_address
            )

            # ------------------------------
            # Name blocking
            # ------------------------------

            for token in candidate_name_tokens:

                queries = name_lookup.get(
                    token
                )

                if not queries:
                    continue

                for s1_id, threshold in queries:

                    if country == query_info[s1_id]["country"]:
                        candidate_sets[
                            (s1_id, threshold)
                        ].add(entity_id)

            # ------------------------------
            # Address blocking
            # ------------------------------

            for token in candidate_address_tokens:

                queries = address_lookup.get(
                    token
                )

                if not queries:
                    continue

                for s1_id, threshold in queries:

                    if country == query_info[s1_id]["country"]:
                        candidate_sets[
                            (s1_id, threshold)
                        ].add(entity_id)


# --------------------------------------------------
# Evaluate against ground truth
# --------------------------------------------------

print("\n")
print("=" * 60)
print("RARE-TOKEN BLOCKING EVALUATION")
print("=" * 60)


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    gt_row = ground_truth[
        ground_truth["source1_entity_id"] == s1_id
    ]

    if gt_row.empty:
        continue

    value = gt_row.iloc[0]["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        true_matches = set()
    else:
        true_matches = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    print(
        f"\nS1: {s1_id}"
    )
    print(
        f"Business: {row['business_name']}"
    )

    for threshold in THRESHOLDS:

        candidates = candidate_sets[
            (s1_id, threshold)
        ]

        captured = (
            true_matches & candidates
        )

        missed = (
            true_matches - candidates
        )

        if true_matches:
            recall = (
                len(captured)
                / len(true_matches)
            )
        else:
            recall = (
                1.0
                if not candidates
                else 0.0
            )

        print(
            f"  threshold={threshold:>3} | "
            f"candidates={len(candidates):>8,} | "
            f"captured={len(captured):>2}/{len(true_matches):<2} | "
            f"recall={recall:.2%}"
        )

        if missed:
            print(
                "     MISSED:",
                sorted(missed)
            )