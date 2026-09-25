import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import json


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessing import normalize_name, normalize_address


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 200_000
SINGLE_TOKEN_MAX_FREQ = 10_000

SOURCE1_PATH = PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv"
SOURCE2_PATH = PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv"
SOURCE3_PATH = PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv"
GROUND_TRUTH_PATH = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"

NAME_TOKEN_COUNTS_PATH = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "data"
    / "token_stats"
    / "name_token_counts.json"
)

ADDRESS_TOKEN_COUNTS_PATH = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "data"
    / "token_stats"
    / "address_token_counts.json"
)


# ============================================================
# DIFFICULT CASES WE WANT TO VERIFY
# ============================================================

TARGET_IDS = [
    "S1-867998778",
    "S1-42246345",
    "S1-174146241",
    "S1-216733182",
    "S1-97176033",
    "S1-983540069",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_tokens(value, normalizer):
    """
    Normalize text and return tokens with length >= 2.

    This matches the current blocking evaluator.
    """
    normalized = normalizer(value)

    if not normalized:
        return []

    return sorted(
        set(
            token
            for token in normalized.split()
            if len(token) >= 2
        )
    )


def compact_text(value):
    """
    Remove non-alphanumeric characters after name normalization.

    Matches the current evaluator.
    """
    normalized = normalize_name(value)

    if not normalized:
        return ""

    return "".join(
        char
        for char in normalized
        if char.isalnum()
    )


def rare_tokens(tokens, counts, max_frequency):
    """
    Return tokens whose global frequency is <= max_frequency.
    """
    return [
        token
        for token in tokens
        if counts.get(token, 0) <= max_frequency
    ]


def top_two_tokens(tokens, counts):
    """
    Pick the two least-frequent tokens.

    Matches the current evaluator.
    """
    if not tokens:
        return []

    sorted_tokens = sorted(
        tokens,
        key=lambda token: counts.get(token, 0)
    )

    return sorted_tokens[:2]


def make_pair(tokens):
    """
    Create an order-independent pair from two tokens.
    """
    if len(tokens) < 2:
        return None

    a, b = sorted(tokens[:2])
    return f"{a}|{b}"


# ============================================================
# LOAD TOKEN STATISTICS
# ============================================================

print("=" * 80)
print("VERIFY BLOCKING CASES")
print("=" * 80)

print("\nLoading token statistics...")

with open(NAME_TOKEN_COUNTS_PATH, "r", encoding="utf-8") as f:
    name_token_counts = json.load(f)

with open(ADDRESS_TOKEN_COUNTS_PATH, "r", encoding="utf-8") as f:
    address_token_counts = json.load(f)

print(f"Name tokens loaded: {len(name_token_counts):,}")
print(f"Address tokens loaded: {len(address_token_counts):,}")


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("\nLoading Source 1...")

source1 = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype=str
).fillna("")

source1["entity_id"] = source1["entity_id"].astype(str)

target_source1 = source1[
    source1["entity_id"].isin(TARGET_IDS)
].copy()

print(f"Source 1 rows: {len(source1):,}")
print(f"Target cases found: {len(target_source1)}")


missing_targets = set(TARGET_IDS) - set(
    target_source1["entity_id"]
)

if missing_targets:
    print("\nWARNING: These target IDs were not found:")
    for entity_id in sorted(missing_targets):
        print("  ", entity_id)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

ground_truth = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str
).fillna("")

ground_truth["source1_entity_id"] = (
    ground_truth["source1_entity_id"]
    .astype(str)
)


# ============================================================
# BUILD TRUE MATCH LOOKUP
# ============================================================

true_matches = {}

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]

    if s1_id not in TARGET_IDS:
        continue

    raw_matches = row["matched_entity_ids"]

    if not raw_matches:
        matches = []
    else:
        matches = [
            x.strip()
            for x in raw_matches.split(",")
            if x.strip()
        ]

    true_matches[s1_id] = set(matches)


# ============================================================
# BUILD QUERY INFORMATION
# ============================================================

print("\nPreparing blocking keys for target S1 entities...")

query_info = {}

for _, row in target_source1.iterrows():

    s1_id = row["entity_id"]

    name_tokens = get_tokens(
        row["business_name"],
        normalize_name
    )

    address_tokens = get_tokens(
        row["business_address"],
        normalize_address
    )

    rare_name = rare_tokens(
        name_tokens,
        name_token_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    rare_address = rare_tokens(
        address_tokens,
        address_token_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    top_name = top_two_tokens(
        name_tokens,
        name_token_counts
    )

    top_address = top_two_tokens(
        address_tokens,
        address_token_counts
    )

    query_info[s1_id] = {
        "country": row["country"],

        "normalized_name": normalize_name(
            row["business_name"]
        ),

        "compact_name": compact_text(
            row["business_name"]
        ),

        "name_tokens": name_tokens,

        "rare_name": rare_name,

        "name_pair": make_pair(top_name),

        "address_tokens": address_tokens,

        "rare_address": rare_address,

        "address_pair": make_pair(top_address),
    }


# ============================================================
# BUILD EXACT SAME BLOCKING INDEXES
# ============================================================

print("\nBuilding blocking indexes from target S1 rows...")

exact_name_lookup = defaultdict(set)
compact_name_lookup = defaultdict(set)
name_single_lookup = defaultdict(set)
name_pair_lookup = defaultdict(set)
address_single_lookup = defaultdict(set)
address_pair_lookup = defaultdict(set)

for s1_id, info in query_info.items():

    # --------------------------------------------------------
    # Exact normalized name
    # --------------------------------------------------------

    normalized_name = info["normalized_name"]

    if normalized_name:
        exact_name_lookup[normalized_name].add(s1_id)

    # --------------------------------------------------------
    # Compact name
    # --------------------------------------------------------

    compact_name = info["compact_name"]

    if compact_name:
        compact_name_lookup[compact_name].add(s1_id)

    # --------------------------------------------------------
    # Rare name tokens
    # --------------------------------------------------------

    for token in info["rare_name"]:
        name_single_lookup[token].add(s1_id)

    # --------------------------------------------------------
    # Name pair
    # --------------------------------------------------------

    name_pair = info["name_pair"]

    if name_pair:
        name_pair_lookup[name_pair].add(s1_id)

    # --------------------------------------------------------
    # Rare address tokens
    # --------------------------------------------------------

    for token in info["rare_address"]:
        address_single_lookup[token].add(s1_id)

    # --------------------------------------------------------
    # Address pair
    # --------------------------------------------------------

    address_pair = info["address_pair"]

    if address_pair:
        address_pair_lookup[address_pair].add(s1_id)


# ============================================================
# FUNCTION TO CHECK ONE SOURCE ROW
# ============================================================

def get_blocking_matches(row):
    """
    Apply the SAME six blocking mechanisms used in
    evaluate_blocking_no_numbers.py.

    Returns:
        matched S1 IDs + rules responsible for each match.
    """

    matches_by_rule = defaultdict(set)

    normalized_name = normalize_name(
        row["business_name"]
    )

    compact_name = compact_text(
        row["business_name"]
    )

    name_tokens = get_tokens(
        row["business_name"],
        normalize_name
    )

    address_tokens = get_tokens(
        row["business_address"],
        normalize_address
    )

    rare_name = rare_tokens(
        name_tokens,
        name_token_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    rare_address = rare_tokens(
        address_tokens,
        address_token_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    top_name = top_two_tokens(
        name_tokens,
        name_token_counts
    )

    top_address = top_two_tokens(
        address_tokens,
        address_token_counts
    )

    name_pair = make_pair(top_name)
    address_pair = make_pair(top_address)

    # --------------------------------------------------------
    # Rule 1: Exact normalized name
    # --------------------------------------------------------

    if normalized_name:
        for s1_id in exact_name_lookup.get(
            normalized_name,
            set()
        ):
            matches_by_rule["exact_name"].add(s1_id)

    # --------------------------------------------------------
    # Rule 2: Compact name
    # --------------------------------------------------------

    if compact_name:
        for s1_id in compact_name_lookup.get(
            compact_name,
            set()
        ):
            matches_by_rule["compact_name"].add(s1_id)

    # --------------------------------------------------------
    # Rule 3: Rare name token
    # --------------------------------------------------------

    for token in rare_name:

        for s1_id in name_single_lookup.get(
            token,
            set()
        ):
            matches_by_rule["rare_name_token"].add(s1_id)

    # --------------------------------------------------------
    # Rule 4: Name pair
    # --------------------------------------------------------

    if name_pair:

        for s1_id in name_pair_lookup.get(
            name_pair,
            set()
        ):
            matches_by_rule["name_pair"].add(s1_id)

    # --------------------------------------------------------
    # Rule 5: Rare address token
    # --------------------------------------------------------

    for token in rare_address:

        for s1_id in address_single_lookup.get(
            token,
            set()
        ):
            matches_by_rule["rare_address_token"].add(s1_id)

    # --------------------------------------------------------
    # Rule 6: Address pair
    # --------------------------------------------------------

    if address_pair:

        for s1_id in address_pair_lookup.get(
            address_pair,
            set()
        ):
            matches_by_rule["address_pair"].add(s1_id)

    return matches_by_rule


# ============================================================
# SCAN SOURCE 2 AND SOURCE 3
# ============================================================

print("\nScanning Source 2 and Source 3...")
print("This may take some time because the files are large.")


# For every true match, store:
#
# candidate ID -> set of rules that captured it
#
captured_matches = {
    s1_id: defaultdict(set)
    for s1_id in TARGET_IDS
}


def process_source_file(source_path, source_label):

    print(f"\nProcessing {source_label}: {source_path.name}")

    rows_processed = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ).fillna(""):

        rows_processed += len(chunk)

        for _, row in chunk.iterrows():

            source_entity_id = row["entity_id"]

            # ------------------------------------------------
            # We only care about source rows that appear in
            # the ground truth of our six target entities.
            # ------------------------------------------------

            relevant_s1_ids = []

            for s1_id in TARGET_IDS:

                if source_entity_id in true_matches.get(
                    s1_id,
                    set()
                ):
                    relevant_s1_ids.append(s1_id)

            if not relevant_s1_ids:
                continue

            # ------------------------------------------------
            # Apply EXACT blocking logic
            # ------------------------------------------------

            matches_by_rule = get_blocking_matches(row)

            # ------------------------------------------------
            # Country filtering
            #
            # IMPORTANT:
            # Country is NOT a blocking key.
            #
            # It is applied AFTER a blocking rule produces
            # an S1 candidate, exactly like the evaluator.
            # ------------------------------------------------

            source_country = row["country"]

            for rule, candidate_s1_ids in matches_by_rule.items():

                for s1_id in candidate_s1_ids:

                    if (
                        query_info[s1_id]["country"]
                        == source_country
                    ):
                        captured_matches[s1_id][
                            source_entity_id
                        ].add(rule)

        print(
            f"  Processed {rows_processed:,} rows...",
            flush=True
        )


process_source_file(
    SOURCE2_PATH,
    "Source 2"
)

process_source_file(
    SOURCE3_PATH,
    "Source 3"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 80)
print("FINAL VERIFICATION REPORT")
print("=" * 80)


for s1_id in TARGET_IDS:

    print("\n" + "-" * 80)
    print(f"S1 ENTITY: {s1_id}")
    print("-" * 80)

    info = query_info.get(s1_id)

    if not info:
        print("S1 entity not found.")
        continue

    print(
        f"Country: {info['country']}"
    )

    print(
        f"Normalized name: {info['normalized_name']}"
    )

    print(
        f"True matches: {len(true_matches.get(s1_id, set()))}"
    )

    print(
        f"Captured true matches: "
        f"{len(captured_matches[s1_id])}"
    )

    missed = (
        true_matches.get(s1_id, set())
        - set(captured_matches[s1_id].keys())
    )

    captured = (
        true_matches.get(s1_id, set())
        & set(captured_matches[s1_id].keys())
    )

    print(
        f"Missed true matches: {len(missed)}"
    )

    # --------------------------------------------------------
    # Captured matches
    # --------------------------------------------------------

    if captured:

        print("\nCAPTURED TRUE MATCHES:")

        for entity_id in sorted(captured):

            rules = sorted(
                captured_matches[s1_id][entity_id]
            )

            print(
                f"  YES  {entity_id}"
                f"  -> {', '.join(rules)}"
            )

    # --------------------------------------------------------
    # Missed matches
    # --------------------------------------------------------

    if missed:

        print("\nMISSED TRUE MATCHES:")

        for entity_id in sorted(missed):

            print(
                f"  NO   {entity_id}"
                f"  -> no blocking rule captured it"
            )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    total_true = len(
        true_matches.get(s1_id, set())
    )

    if total_true > 0:

        recall = (
            len(captured)
            / total_true
        )

        print(
            f"\nBlocking recall for {s1_id}: "
            f"{recall:.4f} "
            f"({len(captured)}/{total_true})"
        )


# ============================================================
# OVERALL SUMMARY
# ============================================================

total_true_matches = 0
total_captured_matches = 0

print("\n")
print("=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)

for s1_id in TARGET_IDS:

    true_count = len(
        true_matches.get(s1_id, set())
    )

    captured_count = len(
        captured_matches[s1_id]
    )

    total_true_matches += true_count
    total_captured_matches += captured_count

    print(
        f"{s1_id}: "
        f"{captured_count}/{true_count} captured"
    )


if total_true_matches > 0:

    overall_recall = (
        total_captured_matches
        / total_true_matches
    )

    print("\nOverall blocking recall:")
    print(
        f"{overall_recall:.4f} "
        f"({total_captured_matches}/"
        f"{total_true_matches})"
    )


print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)