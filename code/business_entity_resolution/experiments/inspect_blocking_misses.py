import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "src"
)

sys.path.insert(0, str(SRC_DIR))

from src.preprocessing import normalize_name, normalize_address


# =================================================
# Configuration
# =================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset" / "train"

S1_PATH = DATASET_DIR / "train_source1.tsv"
S2_PATH = DATASET_DIR / "train_source2.tsv"
S3_PATH = DATASET_DIR / "train_source3.tsv"
GROUND_TRUTH_PATH = DATASET_DIR / "train_ground_truth.tsv"

NAME_STATS_PATH = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "data"
    / "token_stats"
    / "name_token_counts.json"
)

ADDRESS_STATS_PATH = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "data"
    / "token_stats"
    / "address_token_counts.json"
)


# These are the six difficult cases from
# BLOCKING V2 WITHOUT NUMERIC BLOCKING.

DIFFICULT_IDS = [
    "S1-867998778",
    "S1-42246345",
    "S1-174146241",
    "S1-216733182",
    "S1-97176033",
    "S1-983540069",
]


SINGLE_TOKEN_MAX_FREQ = 10_000


# =================================================
# Helper Functions
# =================================================

def safe_text(value):
    """
    Convert a value to a clean string.
    Missing values become an empty string.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


def get_tokens(value, normalizer):
    """
    Normalize text and return unique tokens.

    One-character tokens are ignored for blocking,
    matching the existing V2 evaluator.
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
    Create compact normalized name.

    Example:
        Prime Money
        ->
        primemoney
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
    Select tokens whose frequency is
    below the configured maximum.
    """

    return [
        token
        for token in tokens
        if counts.get(
            token,
            10**18
        ) <= max_frequency
    ]


def top_two_tokens(
    tokens,
    counts
):
    """
    Select the two least frequent tokens.
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
# Blocking Key Extraction
# =================================================

def extract_blocking_keys(
    name,
    address,
    name_counts,
    address_counts
):
    """
    Extract exactly the same blocking keys
    used in V2 WITHOUT numeric blocking.
    """

    # ---------------------------------------------
    # Name
    # ---------------------------------------------

    normalized_name = normalize_name(name)

    compact_name = compact_text(name)

    name_tokens = get_tokens(
        name,
        normalize_name
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
    # Address
    # ---------------------------------------------

    normalized_address = normalize_address(
        address
    )

    address_tokens = get_tokens(
        address,
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

    return {
        "normalized_name": normalized_name,
        "compact_name": compact_name,
        "name_tokens": name_tokens,
        "name_single": set(name_single),
        "name_pair": name_pair,
        "normalized_address": normalized_address,
        "address_tokens": address_tokens,
        "address_single": set(address_single),
        "address_pair": address_pair,
    }


# =================================================
# Load Token Statistics
# =================================================

print("=" * 70)
print("LOADING TOKEN STATISTICS")
print("=" * 70)

with open(
    NAME_STATS_PATH,
    "r",
    encoding="utf-8"
) as f:

    name_counts = json.load(f)


with open(
    ADDRESS_STATS_PATH,
    "r",
    encoding="utf-8"
) as f:

    address_counts = json.load(f)


print(
    f"Name tokens loaded: "
    f"{len(name_counts):,}"
)

print(
    f"Address tokens loaded: "
    f"{len(address_counts):,}"
)


# =================================================
# Load Source 1
# =================================================

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t"
)

print(
    f"Source 1 records: "
    f"{len(s1):,}"
)


# =================================================
# Load Ground Truth
# =================================================

print("\nLoading Ground Truth...")

ground_truth = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t"
)

print(
    f"Ground Truth records: "
    f"{len(ground_truth):,}"
)


# =================================================
# Build Ground Truth Lookup
# =================================================

gt_lookup = {}

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]

    value = row["matched_entity_ids"]

    if (
        pd.isna(value)
        or str(value).strip() == ""
    ):

        matches = set()

    else:

        matches = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    gt_lookup[s1_id] = matches


# =================================================
# Select Difficult Source 1 Records
# =================================================

s1_difficult = s1[
    s1["entity_id"].isin(DIFFICULT_IDS)
].copy()


print("\n")
print("=" * 70)
print("DIFFICULT SOURCE 1 RECORDS FOUND")
print("=" * 70)

print(
    f"Requested: {len(DIFFICULT_IDS)}"
)

print(
    f"Found:     {len(s1_difficult)}"
)

missing_difficult = (
    set(DIFFICULT_IDS)
    - set(s1_difficult["entity_id"])
)

if missing_difficult:

    print("\nWARNING: These IDs were not found:")

    for s1_id in sorted(missing_difficult):

        print(
            f"  {s1_id}"
        )


# =================================================
# Build S1 Blocking Information
# =================================================

query_info = {}

exact_name_lookup = defaultdict(set)
compact_name_lookup = defaultdict(set)

name_single_lookup = defaultdict(set)
name_pair_lookup = defaultdict(set)

address_single_lookup = defaultdict(set)
address_pair_lookup = defaultdict(set)


print("\nBuilding Source 1 blocking indexes...")


for _, row in s1_difficult.iterrows():

    s1_id = row["entity_id"]

    country = (
        safe_text(row["country"])
        .casefold()
    )

    keys = extract_blocking_keys(
        row["business_name"],
        row["business_address"],
        name_counts,
        address_counts
    )

    query_info[s1_id] = {
        "country": country,
        "keys": keys
    }

    # Exact normalized name

    if keys["normalized_name"]:

        exact_name_lookup[
            keys["normalized_name"]
        ].add(s1_id)

    # Compact name

    if keys["compact_name"]:

        compact_name_lookup[
            keys["compact_name"]
        ].add(s1_id)

    # Rare name tokens

    for token in keys["name_single"]:

        name_single_lookup[
            token
        ].add(s1_id)

    # Name pair

    if keys["name_pair"]:

        name_pair_lookup[
            keys["name_pair"]
        ].add(s1_id)

    # Rare address tokens

    for token in keys["address_single"]:

        address_single_lookup[
            token
        ].add(s1_id)

    # Address pair

    if keys["address_pair"]:

        address_pair_lookup[
            keys["address_pair"]
        ].add(s1_id)


# =================================================
# Find Candidate Blocking Reasons
# =================================================

def check_blocking_reasons(
    s1_id,
    candidate_row
):
    """
    Check which individual blocking rules
    connect the Source 1 record to a candidate.
    """

    s1_data = query_info[s1_id]

    s1_keys = s1_data["keys"]

    candidate_name = candidate_row[
        "business_name"
    ]

    candidate_address = candidate_row[
        "business_address"
    ]

    candidate_country = (
        safe_text(
            candidate_row["country"]
        ).casefold()
    )

    candidate_keys = extract_blocking_keys(
        candidate_name,
        candidate_address,
        name_counts,
        address_counts
    )

    reasons = []

    # ---------------------------------------------
    # Country
    # ---------------------------------------------

    same_country = (
        s1_data["country"]
        == candidate_country
    )

    if same_country:
        reasons.append("country")

    # ---------------------------------------------
    # Exact normalized name
    # ---------------------------------------------

    if (
        s1_keys["normalized_name"]
        and
        s1_keys["normalized_name"]
        == candidate_keys["normalized_name"]
    ):

        reasons.append(
            "exact_normalized_name"
        )

    # ---------------------------------------------
    # Compact name
    # ---------------------------------------------

    if (
        s1_keys["compact_name"]
        and
        s1_keys["compact_name"]
        == candidate_keys["compact_name"]
    ):

        reasons.append(
            "compact_name"
        )

    # ---------------------------------------------
    # Rare name token
    # ---------------------------------------------

    common_name_tokens = (
        s1_keys["name_single"]
        &
        candidate_keys["name_single"]
    )

    if common_name_tokens:

        reasons.append(
            "rare_name_token"
        )

    # ---------------------------------------------
    # Name pair
    # ---------------------------------------------

    if (
        s1_keys["name_pair"]
        and
        s1_keys["name_pair"]
        == candidate_keys["name_pair"]
    ):

        reasons.append(
            "name_token_pair"
        )

    # ---------------------------------------------
    # Rare address token
    # ---------------------------------------------

    common_address_tokens = (
        s1_keys["address_single"]
        &
        candidate_keys["address_single"]
    )

    if common_address_tokens:

        reasons.append(
            "rare_address_token"
        )

    # ---------------------------------------------
    # Address pair
    # ---------------------------------------------

    if (
        s1_keys["address_pair"]
        and
        s1_keys["address_pair"]
        == candidate_keys["address_pair"]
    ):

        reasons.append(
            "address_token_pair"
        )

    return {
        "reasons": reasons,
        "same_country": same_country,
        "common_name_tokens": common_name_tokens,
        "common_address_tokens": common_address_tokens,
        "s1_keys": s1_keys,
        "candidate_keys": candidate_keys,
    }


# =================================================
# Process Source 2 + Source 3
# =================================================

source_paths = [
    S2_PATH,
    S3_PATH
]


# For each difficult S1 entity we store
# every true match that we encounter.

true_match_rows = defaultdict(list)


print("\n")
print("=" * 70)
print("SEARCHING SOURCE 2 + SOURCE 3")
print("=" * 70)


for source_path in source_paths:

    print(
        f"\nProcessing: {source_path}"
    )

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            source_path,
            sep="\t",
            chunksize=200_000
        ),
        start=1
    ):

        print(
            f"  Chunk {chunk_number}"
        )

        for _, row in chunk.iterrows():

            entity_id = row["entity_id"]

            # Check whether this entity is a
            # known ground-truth match for any
            # difficult Source 1 record.

            for s1_id in DIFFICULT_IDS:

                if s1_id not in query_info:
                    continue

                true_matches = gt_lookup.get(
                    s1_id,
                    set()
                )

                if entity_id in true_matches:

                    true_match_rows[
                        s1_id
                    ].append(
                        row.copy()
                    )


# =================================================
# Detailed Analysis
# =================================================

print("\n")
print("=" * 70)
print("DETAILED ANALYSIS OF MISSED MATCHES")
print("=" * 70)


for s1_id in DIFFICULT_IDS:

    print("\n")
    print("#" * 70)
    print(f"SOURCE 1 ID: {s1_id}")
    print("#" * 70)

    if s1_id not in query_info:

        print(
            "Source 1 record was not found."
        )

        continue

    # ---------------------------------------------
    # Source 1 row
    # ---------------------------------------------

    s1_rows = s1[
        s1["entity_id"]
        == s1_id
    ]

    if s1_rows.empty:

        print(
            "Source 1 record was not found."
        )

        continue

    s1_row = s1_rows.iloc[0]

    print("\nSOURCE 1 RECORD")
    print("-" * 70)

    print(
        f"Entity ID : "
        f"{s1_row['entity_id']}"
    )

    print(
        f"Name      : "
        f"{s1_row['business_name']}"
    )

    print(
        f"Address   : "
        f"{s1_row['business_address']}"
    )

    print(
        f"Country   : "
        f"{s1_row['country']}"
    )

    # ---------------------------------------------
    # S1 blocking keys
    # ---------------------------------------------

    s1_keys = query_info[
        s1_id
    ]["keys"]

    print("\nSOURCE 1 BLOCKING KEYS")
    print("-" * 70)

    print(
        f"Normalized name : "
        f"{s1_keys['normalized_name']}"
    )

    print(
        f"Compact name    : "
        f"{s1_keys['compact_name']}"
    )

    print(
        f"Name tokens     : "
        f"{sorted(s1_keys['name_tokens'])}"
    )

    print(
        f"Rare name       : "
        f"{sorted(s1_keys['name_single'])}"
    )

    print(
        f"Name pair       : "
        f"{s1_keys['name_pair']}"
    )

    print(
        f"Address tokens  : "
        f"{sorted(s1_keys['address_tokens'])}"
    )

    print(
        f"Rare address    : "
        f"{sorted(s1_keys['address_single'])}"
    )

    print(
        f"Address pair    : "
        f"{s1_keys['address_pair']}"
    )

    # ---------------------------------------------
    # Ground truth
    # ---------------------------------------------

    true_matches = gt_lookup.get(
        s1_id,
        set()
    )

    print("\nGROUND TRUTH MATCHES")
    print("-" * 70)

    print(
        f"Number of true matches: "
        f"{len(true_matches)}"
    )

    print(
        ", ".join(
            sorted(true_matches)
        )
    )

    # ---------------------------------------------
    # Analyze each true match
    # ---------------------------------------------

    rows = true_match_rows.get(
        s1_id,
        []
    )

    if not rows:

        print(
            "\nNo true-match rows were found "
            "while scanning Source 2 and Source 3."
        )

        continue

    print("\nTRUE MATCH ANALYSIS")
    print("-" * 70)

    for candidate_row in rows:

        candidate_id = candidate_row[
            "entity_id"
        ]

        analysis = check_blocking_reasons(
            s1_id,
            candidate_row
        )

        reasons = analysis["reasons"]

        print("\n" + "-" * 70)

        print(
            f"TRUE MATCH: {candidate_id}"
        )

        print(
            f"Name    : "
            f"{candidate_row['business_name']}"
        )

        print(
            f"Address : "
            f"{candidate_row['business_address']}"
        )

        print(
            f"Country : "
            f"{candidate_row['country']}"
        )

        print("\nBLOCKING RULES")

        rule_names = [
            (
                "Exact normalized name",
                "exact_normalized_name"
            ),
            (
                "Compact name",
                "compact_name"
            ),
            (
                "Rare name token",
                "rare_name_token"
            ),
            (
                "Name token pair",
                "name_token_pair"
            ),
            (
                "Rare address token",
                "rare_address_token"
            ),
            (
                "Address token pair",
                "address_token_pair"
            ),
        ]

        for display_name, rule_name in rule_names:

            if rule_name in reasons:

                print(
                    f"  [YES] {display_name}"
                )

            else:

                print(
                    f"  [NO ] {display_name}"
                )

        print(
            "\nCountry matches: "
            f"{analysis['same_country']}"
        )

        if analysis["common_name_tokens"]:

            print(
                "Common rare name tokens: "
                f"{sorted(analysis['common_name_tokens'])}"
            )

        else:

            print(
                "Common rare name tokens: NONE"
            )

        if analysis["common_address_tokens"]:

            print(
                "Common rare address tokens: "
                f"{sorted(analysis['common_address_tokens'])}"
            )

        else:

            print(
                "Common rare address tokens: NONE"
            )

        # -----------------------------------------
        # Final blocking conclusion
        # -----------------------------------------

        if (
            analysis["same_country"]
            and reasons
        ):

            print(
                "\nCONCLUSION:"
            )

            print(
                "  This true match shares at least "
                "one blocking key."
            )

            print(
                "  If it was reported as missed by "
                "the evaluator, check the country "
                "or candidate-set construction."
            )

        else:

            print(
                "\nCONCLUSION:"
            )

            print(
                "  This true match does NOT share "
                "any current V2 blocking key."
            )

            print(
                "  This is a genuine blocking miss."
            )


# =================================================
# Finished
# =================================================

print("\n")
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(
    "\nNo blocking rules were changed."
)

print(
    "This script only investigates why the "
    "difficult true matches were missed."
)