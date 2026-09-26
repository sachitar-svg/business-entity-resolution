import json
from collections import defaultdict

import numpy as np
import pandas as pd

from preprocessing import normalize_name, normalize_address


# =================================================
# Configuration
# =================================================

SAMPLE_SIZE = 100
RANDOM_STATE = 42

CHUNK_SIZE = 200_000

# Maximum frequency allowed for a single-token
# blocking key.
SINGLE_TOKEN_MAX_FREQ = 10_000

NAME_STATS_PATH = (
    "code/business_entity_resolution/data/token_stats/"
    "name_token_counts.json"
)

ADDRESS_STATS_PATH = (
    "code/business_entity_resolution/data/token_stats/"
    "address_token_counts.json"
)


# =================================================
# Load token statistics
# =================================================

with open(NAME_STATS_PATH, "r", encoding="utf-8") as f:
    name_counts = json.load(f)

with open(ADDRESS_STATS_PATH, "r", encoding="utf-8") as f:
    address_counts = json.load(f)


# =================================================
# Load Source 1 + Ground Truth
# =================================================

print("Loading Source 1...")

s1_all = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

print(
    f"Total Source 1 records: {len(s1_all):,}"
)


print("Loading Ground Truth...")

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# Take a deterministic sample of 100 Source 1 records
s1_sample = s1_all.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print(
    f"Sampled Source 1 records: {len(s1_sample):,}"
)


# =================================================
# Build Ground Truth Lookup
# =================================================

gt_lookup = {}

for _, row in ground_truth.iterrows():

    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        matches = set()

    else:
        matches = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    gt_lookup[
        row["source1_entity_id"]
    ] = matches


# =================================================
# Helper Functions
# =================================================

def get_tokens(value, normalizer):
    """
    Normalize text and return unique tokens.

    One-character tokens are ignored for blocking.
    """

    normalized = normalizer(value)

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= 2
    }


def compact_text(value):
    """
    Create a compact normalized name.

    Example:
        Prime Money
        -> primemoney
    """

    normalized = normalize_name(value)

    return "".join(
        char
        for char in normalized
        if char.isalnum()
    )


def rare_tokens(
    tokens,
    counts,
    max_frequency
):
    """
    Select tokens whose frequency is below
    the configured maximum.
    """

    return [
        token
        for token in tokens
        if counts.get(token, 10**18)
        <= max_frequency
    ]


def top_two_tokens(
    tokens,
    counts
):
    """
    Select the two least frequent tokens.

    There is no hard frequency limit here.
    """

    ranked = sorted(
        (
            (
                counts.get(
                    token,
                    10**18
                ),
                token
            )
            for token in tokens
        )
    )

    return [
        token
        for _, token in ranked[:2]
    ]


def make_pair(tokens):
    """
    Create an order-independent pair key.
    """

    if len(tokens) < 2:
        return None

    return "|".join(
        sorted(tokens)
    )


# =================================================
# Build Query-Side Blocking Indexes
#
# IMPORTANT:
# Numeric blocking is intentionally NOT included.
# This file is testing V2 WITHOUT numeric blocking.
# =================================================

query_info = {}

exact_name_lookup = defaultdict(set)
compact_name_lookup = defaultdict(set)

name_single_lookup = defaultdict(set)
name_pair_lookup = defaultdict(set)

address_single_lookup = defaultdict(set)
address_pair_lookup = defaultdict(set)


for _, row in s1_sample.iterrows():

    s1_id = row["entity_id"]

    country = (
        str(row["country"])
        .casefold()
        .strip()
    )

    # ---------------------------------------------
    # Name information
    # ---------------------------------------------

    name_tokens = get_tokens(
        row["business_name"],
        normalize_name
    )

    exact_name = normalize_name(
        row["business_name"]
    )

    compact_name = compact_text(
        row["business_name"]
    )

    name_single = rare_tokens(
        name_tokens,
        name_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    name_pair = make_pair(
        top_two_tokens(
            name_tokens,
            name_counts
        )
    )

    # ---------------------------------------------
    # Address information
    # ---------------------------------------------

    address_tokens = get_tokens(
        row["business_address"],
        normalize_address
    )

    address_single = rare_tokens(
        address_tokens,
        address_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    address_pair = make_pair(
        top_two_tokens(
            address_tokens,
            address_counts
        )
    )

    # ---------------------------------------------
    # Store S1 information
    # ---------------------------------------------

    query_info[s1_id] = {
        "country": country
    }

    # ---------------------------------------------
    # Exact normalized name
    # ---------------------------------------------

    if exact_name:

        exact_name_lookup[
            exact_name
        ].add(s1_id)

    # ---------------------------------------------
    # Compact name
    # ---------------------------------------------

    if compact_name:

        compact_name_lookup[
            compact_name
        ].add(s1_id)

    # ---------------------------------------------
    # Rare name tokens
    # ---------------------------------------------

    for token in name_single:

        name_single_lookup[
            token
        ].add(s1_id)

    # ---------------------------------------------
    # Name token pair
    # ---------------------------------------------

    if name_pair:

        name_pair_lookup[
            name_pair
        ].add(s1_id)

    # ---------------------------------------------
    # Rare address tokens
    # ---------------------------------------------

    for token in address_single:

        address_single_lookup[
            token
        ].add(s1_id)

    # ---------------------------------------------
    # Address token pair
    # ---------------------------------------------

    if address_pair:

        address_pair_lookup[
            address_pair
        ].add(s1_id)


# =================================================
# Candidate Storage
# =================================================

candidate_sets = defaultdict(set)


# =================================================
# Process Source 2 + Source 3
#
# Numeric blocking is deliberately disabled.
# =================================================

source_paths = [
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv"
]


for source_path in source_paths:

    print(
        f"\nProcessing: {source_path}"
    )

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

            country = (
                str(row["country"])
                .casefold()
                .strip()
            )

            # -------------------------------------
            # Normalize candidate
            # -------------------------------------

            name = row["business_name"]
            address = row["business_address"]

            normalized_name = normalize_name(
                name
            )

            compact_name = compact_text(
                name
            )

            name_tokens = get_tokens(
                name,
                normalize_name
            )

            address_tokens = get_tokens(
                address,
                normalize_address
            )

            matched_s1 = set()

            # -------------------------------------
            # 1. Exact normalized name
            # -------------------------------------

            if normalized_name:

                matched_s1.update(
                    exact_name_lookup.get(
                        normalized_name,
                        set()
                    )
                )

            # -------------------------------------
            # 2. Compact name
            # -------------------------------------

            if compact_name:

                matched_s1.update(
                    compact_name_lookup.get(
                        compact_name,
                        set()
                    )
                )

            # -------------------------------------
            # 3. Rare name tokens
            # -------------------------------------

            candidate_name_single = rare_tokens(
                name_tokens,
                name_counts,
                SINGLE_TOKEN_MAX_FREQ
            )

            for token in candidate_name_single:

                matched_s1.update(
                    name_single_lookup.get(
                        token,
                        set()
                    )
                )

            # -------------------------------------
            # 4. Name token pair
            # -------------------------------------

            candidate_name_pair = make_pair(
                top_two_tokens(
                    name_tokens,
                    name_counts
                )
            )

            if candidate_name_pair:

                matched_s1.update(
                    name_pair_lookup.get(
                        candidate_name_pair,
                        set()
                    )
                )

            # -------------------------------------
            # 5. Rare address tokens
            # -------------------------------------

            candidate_address_single = rare_tokens(
                address_tokens,
                address_counts,
                SINGLE_TOKEN_MAX_FREQ
            )

            for token in candidate_address_single:

                matched_s1.update(
                    address_single_lookup.get(
                        token,
                        set()
                    )
                )

            # -------------------------------------
            # 6. Address token pair
            # -------------------------------------

            candidate_address_pair = make_pair(
                top_two_tokens(
                    address_tokens,
                    address_counts
                )
            )

            if candidate_address_pair:

                matched_s1.update(
                    address_pair_lookup.get(
                        candidate_address_pair,
                        set()
                    )
                )

            # -------------------------------------
            # 7. Country filter
            # -------------------------------------

            for s1_id in matched_s1:

                if (
                    country
                    == query_info[s1_id]["country"]
                ):

                    candidate_sets[
                        s1_id
                    ].add(entity_id)


# =================================================
# Evaluate Blocking
# =================================================

results = []

print("\n")
print("=" * 70)
print("BLOCKING V2 WITHOUT NUMERIC BLOCKING")
print("=" * 70)


for _, row in s1_sample.iterrows():

    s1_id = row["entity_id"]

    true_matches = gt_lookup.get(
        s1_id,
        set()
    )

    candidates = candidate_sets.get(
        s1_id,
        set()
    )

    captured = (
        true_matches
        & candidates
    )

    missed = (
        true_matches
        - candidates
    )

    # ---------------------------------------------
    # Non-singletons
    # ---------------------------------------------

    if true_matches:

        recall = (
            len(captured)
            / len(true_matches)
        )

        has_true_match = True

    # ---------------------------------------------
    # Singletons
    # ---------------------------------------------

    else:

        recall = None
        has_true_match = False

    # ---------------------------------------------
    # Save result
    # ---------------------------------------------

    results.append({
        "s1_id": s1_id,
        "true_matches": len(true_matches),
        "candidates": len(candidates),
        "captured": len(captured),
        "missed": len(missed),
        "recall": recall,
        "has_true_match": has_true_match
    })


results_df = pd.DataFrame(
    results
)


# =================================================
# Summary
# =================================================

print("\n===== SUMMARY =====")

matched_entities = results_df[
    results_df["has_true_match"]
]

print(
    f"Entities evaluated: "
    f"{len(results_df):,}"
)

print(
    f"Entities with true matches: "
    f"{len(matched_entities):,}"
)

print(
    f"Entities with 100% recall: "
    f"{(
        matched_entities["recall"]
        == 1.0
    ).sum():,}"
)

print(
    f"Entities with missed matches: "
    f"{(
        matched_entities["missed"]
        > 0
    ).sum():,}"
)

if not matched_entities.empty:

    print(
        f"Average blocking recall "
        f"(non-singletons): "
        f"{matched_entities['recall'].mean():.4%}"
    )

else:

    print(
        "Average blocking recall "
        "(non-singletons): N/A"
    )

print(
    f"Mean candidates: "
    f"{results_df['candidates'].mean():,.2f}"
)

print(
    f"Median candidates: "
    f"{results_df['candidates'].median():,.2f}"
)

print(
    f"95th percentile candidates: "
    f"{np.percentile(
        results_df['candidates'],
        95
    ):,.2f}"
)

print(
    f"Maximum candidates: "
    f"{results_df['candidates'].max():,}"
)


# =================================================
# Worst Recall Cases
# =================================================

print("\n===== WORST RECALL CASES =====")

if not matched_entities.empty:

    print(
        matched_entities
        .sort_values(
            ["recall", "candidates"]
        )
        .head(10)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No non-singleton entities found."
    )


# =================================================
# Largest Candidate Sets
# =================================================

print("\n===== LARGEST CANDIDATE SETS =====")

print(
    results_df
    .sort_values(
        "candidates",
        ascending=False
    )
    .head(10)
    .to_string(
        index=False
    )
)