import re
import json
from collections import defaultdict

import pandas as pd

from preprocessing import normalize_name, normalize_address


# -------------------------------------------------
# Configuration
# -------------------------------------------------

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
# Helper functions
# -------------------------------------------------

def get_tokens(value, normalizer):
    """Return normalized tokens."""

    normalized = normalizer(value)

    if not normalized:
        return set()

    return set(normalized.split())


def compact_text(value):
    """
    Remove spaces/punctuation while preserving Unicode
    letters and numbers.
    """

    normalized = normalize_name(value)

    return "".join(
        char
        for char in normalized
        if char.isalnum()
    )


def normalized_numbers(value):
    """
    Extract numbers and remove leading zeros.

    Example:
        0017560 -> 17560
    """

    if not value:
        return set()

    numbers = re.findall(
        r"\d+",
        str(value)
    )

    result = set()

    for number in numbers:

        cleaned = number.lstrip("0")

        if cleaned == "":
            cleaned = "0"

        result.add(cleaned)

    return result


def rare_single_tokens(
    tokens,
    counts,
    max_frequency
):
    """
    Select relatively informative single tokens.
    """

    result = []

    for token in tokens:

        frequency = counts.get(token)

        if frequency is None:
            continue

        if frequency <= max_frequency:
            result.append(token)

    return result


def top_two_tokens(
    tokens,
    counts
):
    """
    Select the two least frequent tokens.

    Unlike the old blocker, there is NO hard frequency
    cutoff for the pair.
    """

    ranked = []

    for token in tokens:

        frequency = counts.get(
            token,
            10**18
        )

        ranked.append(
            (frequency, token)
        )

    ranked.sort()

    return [
        token
        for _, token in ranked[:2]
    ]


def make_pair(tokens):
    """
    Create an order-independent token pair.
    """

    if len(tokens) < 2:
        return None

    return "|".join(
        sorted(tokens)
    )


# -------------------------------------------------
# Load sample Source 1
# -------------------------------------------------

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=5
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# -------------------------------------------------
# Build query-side blocking keys
# -------------------------------------------------

query_info = {}

# Reverse lookup tables.
#
# key -> set(S1 IDs)

exact_name_lookup = defaultdict(set)
compact_name_lookup = defaultdict(set)

name_single_lookup = defaultdict(set)
name_pair_lookup = defaultdict(set)

address_single_lookup = defaultdict(set)
address_pair_lookup = defaultdict(set)

number_lookup = defaultdict(set)


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    country = (
        str(row["country"])
        .casefold()
        .strip()
    )

    name = row["business_name"]
    address = row["business_address"]

    name_tokens = get_tokens(
        name,
        normalize_name
    )

    address_tokens = get_tokens(
        address,
        normalize_address
    )

    # -----------------------------------------
    # Name keys
    # -----------------------------------------

    exact_name = normalize_name(name)

    compact_name = compact_text(name)

    name_rare = rare_single_tokens(
        name_tokens,
        name_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    name_top_two = top_two_tokens(
        name_tokens,
        name_counts
    )

    name_pair = make_pair(
        name_top_two
    )

    # -----------------------------------------
    # Address keys
    # -----------------------------------------

    address_rare = rare_single_tokens(
        address_tokens,
        address_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    address_top_two = top_two_tokens(
        address_tokens,
        address_counts
    )

    address_pair = make_pair(
        address_top_two
    )

    numbers = normalized_numbers(
        address
    )

    query_info[s1_id] = {
        "country": country
    }

    # Reverse indexes

    if exact_name:
        exact_name_lookup[
            exact_name
        ].add(s1_id)

    if compact_name:
        compact_name_lookup[
            compact_name
        ].add(s1_id)

    for token in name_rare:
        name_single_lookup[
            token
        ].add(s1_id)

    if name_pair:
        name_pair_lookup[
            name_pair
        ].add(s1_id)

    for token in address_rare:
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
# Candidate storage
# -------------------------------------------------

candidate_sets = defaultdict(set)

captured_by_channel = defaultdict(
    lambda: defaultdict(set)
)


# -------------------------------------------------
# Process S2 + S3
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

            # -----------------------------
            # Normalize once
            # -----------------------------

            name = row["business_name"]
            address = row["business_address"]

            normalized_name = normalize_name(
                name
            )

            compact = compact_text(
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

            # -----------------------------
            # Name keys
            # -----------------------------

            matched_s1 = set()

            if normalized_name:
                matched_s1.update(
                    exact_name_lookup.get(
                        normalized_name,
                        set()
                    )
                )

            if compact:
                matched_s1.update(
                    compact_name_lookup.get(
                        compact,
                        set()
                    )
                )

            candidate_name_single = (
                rare_single_tokens(
                    name_tokens,
                    name_counts,
                    SINGLE_TOKEN_MAX_FREQ
                )
            )

            for token in candidate_name_single:
                matched_s1.update(
                    name_single_lookup.get(
                        token,
                        set()
                    )
                )

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

            # -----------------------------
            # Address keys
            # -----------------------------

            candidate_address_single = (
                rare_single_tokens(
                    address_tokens,
                    address_counts,
                    SINGLE_TOKEN_MAX_FREQ
                )
            )

            for token in candidate_address_single:
                matched_s1.update(
                    address_single_lookup.get(
                        token,
                        set()
                    )
                )

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

            # -----------------------------
            # Numeric address blocking
            # -----------------------------

            for number in numbers:

                matched_s1.update(
                    number_lookup.get(
                        number,
                        set()
                    )
                )

            # -----------------------------
            # Country filter
            # -----------------------------

            for s1_id in matched_s1:

                if (
                    country
                    != query_info[s1_id]["country"]
                ):
                    continue

                candidate_sets[
                    s1_id
                ].add(entity_id)


# -------------------------------------------------
# Evaluate against ground truth
# -------------------------------------------------

print("\n")
print("=" * 70)
print("BLOCKING V2 EVALUATION")
print("=" * 70)


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    gt_row = ground_truth[
        ground_truth["source1_entity_id"]
        == s1_id
    ]

    if gt_row.empty:
        continue

    value = (
        gt_row.iloc[0]["matched_entity_ids"]
    )

    if pd.isna(value) or str(value).strip() == "":
        true_matches = set()
    else:
        true_matches = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    candidates = candidate_sets[
        s1_id
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

    print("\n------------------------------------------")
    print("S1:", s1_id)
    print(
        "Business:",
        row["business_name"]
    )

    print(
        "True matches:",
        len(true_matches)
    )

    print(
        "Candidates:",
        len(candidates)
    )

    print(
        "Captured:",
        len(captured)
    )

    print(
        "Missed:",
        len(missed)
    )

    print(
        "Blocking recall:",
        f"{recall:.2%}"
    )

    if missed:
        print(
            "\nMISSED TRUE MATCHES:"
        )

        for entity_id in sorted(missed):
            print(
                " ",
                entity_id
            )