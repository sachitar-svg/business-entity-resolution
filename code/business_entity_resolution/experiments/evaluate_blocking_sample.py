import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from preprocessing import normalize_name, normalize_address


# -------------------------------------------------
# Configuration
# -------------------------------------------------

SAMPLE_SIZE = 100
RANDOM_STATE = 42

CHUNK_SIZE = 200_000

SINGLE_TOKEN_MAX_FREQ = 10_000

NAME_STATS_PATH = (
    "code/business_entity_resolution/data/token_stats/"
    "name_token_counts.json"
)

ADDRESS_STATS_PATH = (
    "code/business_entity_resolution/data/token_stats/"
    "address_token_counts.json"
)


# -------------------------------------------------
# Load token statistics
# -------------------------------------------------

with open(NAME_STATS_PATH, "r", encoding="utf-8") as f:
    name_counts = json.load(f)

with open(ADDRESS_STATS_PATH, "r", encoding="utf-8") as f:
    address_counts = json.load(f)


# -------------------------------------------------
# Load all Source 1 IDs and take deterministic sample
# -------------------------------------------------

s1_all = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

s1_sample = s1_all.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# -------------------------------------------------
# Ground truth lookup
# -------------------------------------------------

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


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def get_tokens(value, normalizer):

    normalized = normalizer(value)

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= 2
    }


def compact_text(value):

    normalized = normalize_name(value)

    return "".join(
        char
        for char in normalized
        if char.isalnum()
    )


def normalized_numbers(value):

    if value is None or pd.isna(value):
        return set()

    numbers = re.findall(
        r"\d+",
        str(value)
    )

    result = set()

    for number in numbers:

        number = number.lstrip("0")

        if number == "":
            number = "0"

        result.add(number)

    return result


def rare_tokens(
    tokens,
    counts,
    max_frequency
):

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

    if len(tokens) < 2:
        return None

    return "|".join(
        sorted(tokens)
    )


# -------------------------------------------------
# Build query indexes
# -------------------------------------------------

query_info = {}

exact_name_lookup = defaultdict(set)
compact_name_lookup = defaultdict(set)

name_single_lookup = defaultdict(set)
name_pair_lookup = defaultdict(set)

address_single_lookup = defaultdict(set)
address_pair_lookup = defaultdict(set)

number_lookup = defaultdict(set)


for _, row in s1_sample.iterrows():

    s1_id = row["entity_id"]

    country = (
        str(row["country"])
        .casefold()
        .strip()
    )

    name_tokens = get_tokens(
        row["business_name"],
        normalize_name
    )

    address_tokens = get_tokens(
        row["business_address"],
        normalize_address
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

    numbers = normalized_numbers(
        row["business_address"]
    )

    query_info[s1_id] = {
        "country": country
    }

    if exact_name:
        exact_name_lookup[
            exact_name
        ].add(s1_id)

    if compact_name:
        compact_name_lookup[
            compact_name
        ].add(s1_id)

    for token in name_single:
        name_single_lookup[
            token
        ].add(s1_id)

    if name_pair:
        name_pair_lookup[
            name_pair
        ].add(s1_id)

    for token in address_single:
        address_single_lookup[
            token
        ].add(s1_id)

    if address_pair:
        address_pair_lookup[
            address_pair
        ].add(s1_id)

    for number in numbers:
        number_lookup[
            number
        ].add(s1_id)


# -------------------------------------------------
# Candidate sets
# -------------------------------------------------

candidate_sets = defaultdict(set)


# -------------------------------------------------
# Scan S2 + S3
# -------------------------------------------------

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

            numbers = normalized_numbers(
                address
            )

            matched_s1 = set()

            # -------------------------------------
            # Name exact
            # -------------------------------------

            if normalized_name:

                matched_s1.update(
                    exact_name_lookup.get(
                        normalized_name,
                        set()
                    )
                )

            # -------------------------------------
            # Compact name
            # -------------------------------------

            if compact_name:

                matched_s1.update(
                    compact_name_lookup.get(
                        compact_name,
                        set()
                    )
                )

            # -------------------------------------
            # Name single token
            # -------------------------------------

            for token in rare_tokens(
                name_tokens,
                name_counts,
                SINGLE_TOKEN_MAX_FREQ
            ):

                matched_s1.update(
                    name_single_lookup.get(
                        token,
                        set()
                    )
                )

            # -------------------------------------
            # Name pair
            # -------------------------------------

            name_pair = make_pair(
                top_two_tokens(
                    name_tokens,
                    name_counts
                )
            )

            if name_pair:

                matched_s1.update(
                    name_pair_lookup.get(
                        name_pair,
                        set()
                    )
                )

            # -------------------------------------
            # Address single
            # -------------------------------------

            for token in rare_tokens(
                address_tokens,
                address_counts,
                SINGLE_TOKEN_MAX_FREQ
            ):

                matched_s1.update(
                    address_single_lookup.get(
                        token,
                        set()
                    )
                )

            # -------------------------------------
            # Address pair
            # -------------------------------------

            address_pair = make_pair(
                top_two_tokens(
                    address_tokens,
                    address_counts
                )
            )

            if address_pair:

                matched_s1.update(
                    address_pair_lookup.get(
                        address_pair,
                        set()
                    )
                )

            # -------------------------------------
            # Numeric address
            # -------------------------------------

            for number in numbers:

                matched_s1.update(
                    number_lookup.get(
                        number,
                        set()
                    )
                )

            # -------------------------------------
            # Apply country filter
            # -------------------------------------

            for s1_id in matched_s1:

                if (
                    country
                    == query_info[s1_id]["country"]
                ):

                    candidate_sets[
                        s1_id
                    ].add(entity_id)


# -------------------------------------------------
# Evaluate
# -------------------------------------------------

results = []

print("\n")
print("=" * 70)
print("BLOCKING V2 — 100 ENTITY EVALUATION")
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

    results.append({
        "s1_id": s1_id,
        "true_matches": len(true_matches),
        "candidates": len(candidates),
        "captured": len(captured),
        "missed": len(missed),
        "recall": recall
    })


results_df = pd.DataFrame(results)


# -------------------------------------------------
# Summary
# -------------------------------------------------

print("\n===== SUMMARY =====")

print(
    f"Entities evaluated: "
    f"{len(results_df):,}"
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
    f"Minimum candidates: "
    f"{results_df['candidates'].min():,}"
)

print(
    f"Maximum candidates: "
    f"{results_df['candidates'].max():,}"
)

print(
    f"95th percentile candidates: "
    f"{np.percentile(results_df['candidates'], 95):,.2f}"
)

print(
    f"Entities with 100% recall: "
    f"{(results_df['recall'] == 1.0).sum():,}"
)

print(
    f"Entities with missed matches: "
    f"{(results_df['missed'] > 0).sum():,}"
)

print(
    f"Average blocking recall: "
    f"{results_df['recall'].mean():.4%}"
)

print("\n===== WORST RECALL CASES =====")

print(
    results_df
    .sort_values(
        ["recall", "candidates"]
    )
    .head(10)
    .to_string(index=False)
)

print("\n===== LARGEST CANDIDATE SETS =====")

print(
    results_df
    .sort_values(
        "candidates",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)