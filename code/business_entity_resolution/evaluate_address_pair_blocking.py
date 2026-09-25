import sys
import re
import json
from pathlib import Path
from itertools import combinations

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


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 100
RANDOM_STATE = 42
CHUNK_SIZE = 200_000

SINGLE_TOKEN_MAX_FREQ = 10_000

# NEW EXPERIMENT:
# maximum frequency allowed for a token used in an address pair
PAIR_TOKEN_MAX_FREQ = 10_000


# ============================================================
# HELPERS
# ============================================================

def get_tokens(value, normalizer):
    value = normalizer(value)

    if not value:
        return set()

    return {
        token
        for token in value.split()
        if len(token) >= 2
    }


def compact_text(value):
    normalized = normalize_name(value)

    return "".join(
        ch for ch in normalized
        if ch.isalnum()
    )


def rare_tokens(tokens, counts, max_frequency):
    return {
        token
        for token in tokens
        if counts.get(token, 0) <= max_frequency
    }


def make_pair(token1, token2):
    return "|".join(sorted([token1, token2]))


def useful_address_pairs(tokens, counts):
    """
    NEW RULE:
    Generate multiple address-token pairs, but only from
    tokens whose frequency is <= PAIR_TOKEN_MAX_FREQ.

    This prevents extremely common tokens from generating
    huge candidate sets.
    """

    useful_tokens = [
        token
        for token in tokens
        if counts.get(token, 0) <= PAIR_TOKEN_MAX_FREQ
    ]

    return {
        make_pair(a, b)
        for a, b in combinations(sorted(useful_tokens), 2)
    }


# ============================================================
# LOAD TOKEN STATISTICS
# ============================================================

print("=" * 80)
print("LOADING TOKEN STATISTICS")
print("=" * 80)

with open(NAME_STATS_PATH, "r", encoding="utf-8") as f:
    name_counts = json.load(f)

with open(ADDRESS_STATS_PATH, "r", encoding="utf-8") as f:
    address_counts = json.load(f)

print(f"Name tokens loaded   : {len(name_counts):,}")
print(f"Address tokens loaded: {len(address_counts):,}")


# ============================================================
# LOAD SOURCE 1 + GROUND TRUTH
# ============================================================

print()
print("=" * 80)
print("LOADING SOURCE 1 AND GROUND TRUTH")
print("=" * 80)

source1 = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype=str
).fillna("")

ground_truth = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str
).fillna("")


# Ground truth lookup
gt_lookup = {}

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]

    matched = str(row["matched_entity_ids"]).strip()

    if not matched:
        gt_lookup[s1_id] = set()
    else:
        gt_lookup[s1_id] = {
            x.strip()
            for x in matched.split(",")
            if x.strip()
        }


# ============================================================
# SAMPLE SOURCE 1
# ============================================================

sample_s1 = source1.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
)

sample_ids = set(sample_s1["entity_id"])

print(f"Sample size: {len(sample_s1)}")


# ============================================================
# BUILD SOURCE 1 BLOCKING INDEXES
# ============================================================

print()
print("=" * 80)
print("BUILDING SOURCE 1 BLOCKING INDEXES")
print("=" * 80)


exact_name_lookup = {}
compact_name_lookup = {}
name_single_lookup = {}
name_pair_lookup = {}

address_single_lookup = {}

# CURRENT address-pair index
address_pair_lookup_current = {}

# NEW multiple-address-pair index
address_pair_lookup_new = {}

query_info = {}


def add_to_index(index, key, s1_id):

    if key not in index:
        index[key] = set()

    index[key].add(s1_id)


for _, row in sample_s1.iterrows():

    s1_id = row["entity_id"]

    name = row["business_name"]
    address = row["business_address"]
    country = row["country"]

    name_norm = normalize_name(name)
    address_norm = normalize_address(address)

    name_tokens = get_tokens(
        name,
        normalize_name
    )

    address_tokens = get_tokens(
        address,
        normalize_address
    )

    # --------------------------------------------------------
    # Store country
    # --------------------------------------------------------

    query_info[s1_id] = {
        "country": country
    }

    # --------------------------------------------------------
    # Name indexes
    # --------------------------------------------------------

    add_to_index(
        exact_name_lookup,
        name_norm,
        s1_id
    )

    add_to_index(
        compact_name_lookup,
        compact_text(name),
        s1_id
    )

    rare_name = rare_tokens(
        name_tokens,
        name_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    for token in rare_name:
        add_to_index(
            name_single_lookup,
            token,
            s1_id
        )

    # Existing name pair
    if len(name_tokens) >= 2:

        sorted_tokens = sorted(
            name_tokens,
            key=lambda x: name_counts.get(x, 10**18)
        )

        pair = make_pair(
            sorted_tokens[0],
            sorted_tokens[1]
        )

        add_to_index(
            name_pair_lookup,
            pair,
            s1_id
        )

    # --------------------------------------------------------
    # Address single-token index
    # --------------------------------------------------------

    rare_address = rare_tokens(
        address_tokens,
        address_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    for token in rare_address:

        add_to_index(
            address_single_lookup,
            token,
            s1_id
        )

    # --------------------------------------------------------
    # CURRENT address pair
    # --------------------------------------------------------

    if len(address_tokens) >= 2:

        sorted_address_tokens = sorted(
            address_tokens,
            key=lambda x: address_counts.get(x, 10**18)
        )

        current_pair = make_pair(
            sorted_address_tokens[0],
            sorted_address_tokens[1]
        )

        add_to_index(
            address_pair_lookup_current,
            current_pair,
            s1_id
        )

    # --------------------------------------------------------
    # NEW: MULTIPLE ADDRESS PAIRS
    # --------------------------------------------------------

    new_pairs = useful_address_pairs(
        address_tokens,
        address_counts
    )

    for pair in new_pairs:

        add_to_index(
            address_pair_lookup_new,
            pair,
            s1_id
        )


print(
    f"Current address pairs : "
    f"{len(address_pair_lookup_current):,}"
)

print(
    f"New address pairs     : "
    f"{len(address_pair_lookup_new):,}"
)


# ============================================================
# EVALUATION STORAGE
# ============================================================

results = []


def get_candidate_ids(row, use_new_address_pairs):
    """
    Apply the blocking rules to one Source 2/3 row.
    """

    candidates = set()

    name = row["business_name"]
    address = row["business_address"]

    name_norm = normalize_name(name)
    address_tokens = get_tokens(
        address,
        normalize_address
    )

    name_tokens = get_tokens(
        name,
        normalize_name
    )

    # --------------------------------------------------------
    # Existing name blocking
    # --------------------------------------------------------

    candidates.update(
        exact_name_lookup.get(
            name_norm,
            set()
        )
    )

    candidates.update(
        compact_name_lookup.get(
            compact_text(name),
            set()
        )
    )

    rare_name = rare_tokens(
        name_tokens,
        name_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    for token in rare_name:

        candidates.update(
            name_single_lookup.get(
                token,
                set()
            )
        )

    if len(name_tokens) >= 2:

        sorted_tokens = sorted(
            name_tokens,
            key=lambda x: name_counts.get(x, 10**18)
        )

        pair = make_pair(
            sorted_tokens[0],
            sorted_tokens[1]
        )

        candidates.update(
            name_pair_lookup.get(
                pair,
                set()
            )
        )

    # --------------------------------------------------------
    # Existing address single-token blocking
    # --------------------------------------------------------

    rare_address = rare_tokens(
        address_tokens,
        address_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    for token in rare_address:

        candidates.update(
            address_single_lookup.get(
                token,
                set()
            )
        )

    # --------------------------------------------------------
    # Address pair blocking
    # --------------------------------------------------------

    if use_new_address_pairs:

        pairs = useful_address_pairs(
            address_tokens,
            address_counts
        )

        for pair in pairs:

            candidates.update(
                address_pair_lookup_new.get(
                    pair,
                    set()
                )
            )

    else:

        if len(address_tokens) >= 2:

            sorted_address_tokens = sorted(
                address_tokens,
                key=lambda x: address_counts.get(x, 10**18)
            )

            current_pair = make_pair(
                sorted_address_tokens[0],
                sorted_address_tokens[1]
            )

            candidates.update(
                address_pair_lookup_current.get(
                    current_pair,
                    set()
                )
            )

    # --------------------------------------------------------
    # Country filter
    # --------------------------------------------------------

    country = row["country"]

    candidates = {
        s1_id
        for s1_id in candidates
        if query_info[s1_id]["country"] == country
    }

    return candidates


# ============================================================
# SCAN SOURCE 2 AND SOURCE 3
# ============================================================

def evaluate_source(source_path, source_name):

    print()
    print("=" * 80)
    print(f"SCANNING {source_name}")
    print("=" * 80)

    reader = pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        chunk = chunk.fillna("")

        print(
            f"Processing chunk {chunk_number}..."
        )

        for _, row in chunk.iterrows():

            source_id = row["entity_id"]

            # ------------------------------------------------
            # Current blocking
            # ------------------------------------------------

            current_candidates = get_candidate_ids(
                row,
                use_new_address_pairs=False
            )

            # ------------------------------------------------
            # New blocking
            # ------------------------------------------------

            new_candidates = get_candidate_ids(
                row,
                use_new_address_pairs=True
            )

            # ------------------------------------------------
            # Evaluate only sampled S1 entities
            # ------------------------------------------------

            for s1_id in sample_ids:

                true_matches = gt_lookup.get(
                    s1_id,
                    set()
                )

                if source_id not in true_matches:
                    continue

                current_hit = s1_id in current_candidates
                new_hit = s1_id in new_candidates

                results.append({
                    "s1_id": s1_id,
                    "true_source_id": source_id,
                    "source": source_name,
                    "current_hit": current_hit,
                    "new_hit": new_hit,
                    "current_candidate_count": len(
                        current_candidates
                    ),
                    "new_candidate_count": len(
                        new_candidates
                    ),
                })


evaluate_source(
    SOURCE2_PATH,
    "Source 2"
)

evaluate_source(
    SOURCE3_PATH,
    "Source 3"
)


# ============================================================
# ANALYSIS
# ============================================================

print()
print("=" * 80)
print("ADDRESS-PAIR BLOCKING EXPERIMENT")
print("=" * 80)

results_df = pd.DataFrame(results)

if results_df.empty:

    print("No ground-truth matches were found in the sample.")
    sys.exit(0)


# ------------------------------------------------------------
# Match recall
# ------------------------------------------------------------

current_recall = results_df["current_hit"].mean()
new_recall = results_df["new_hit"].mean()

print()
print("TRUE MATCH CAPTURE")
print("-" * 80)

print(
    f"Current blocking recall : "
    f"{current_recall * 100:.2f}%"
)

print(
    f"New blocking recall     : "
    f"{new_recall * 100:.2f}%"
)


# ------------------------------------------------------------
# How many additional true matches?
# ------------------------------------------------------------

newly_recovered = (
    (~results_df["current_hit"])
    &
    (results_df["new_hit"])
).sum()

print(
    f"\nNewly recovered matches: "
    f"{newly_recovered}"
)


# ------------------------------------------------------------
# Candidate sizes
# ------------------------------------------------------------

print()
print("CANDIDATE SET SIZE")
print("-" * 80)

print(
    f"Current mean candidates : "
    f"{results_df['current_candidate_count'].mean():.2f}"
)

print(
    f"New mean candidates     : "
    f"{results_df['new_candidate_count'].mean():.2f}"
)

print(
    f"Current median          : "
    f"{results_df['current_candidate_count'].median():.2f}"
)

print(
    f"New median              : "
    f"{results_df['new_candidate_count'].median():.2f}"
)

print(
    f"Current maximum         : "
    f"{results_df['current_candidate_count'].max():,}"
)

print(
    f"New maximum             : "
    f"{results_df['new_candidate_count'].max():,}"
)


# ============================================================
# SHOW THE PREVIOUSLY MISSED MATCHES
# ============================================================

print()
print("=" * 80)
print("PREVIOUSLY MISSED MATCHES")
print("=" * 80)

previously_missed = results_df[
    results_df["current_hit"] == False
]

if previously_missed.empty:

    print("No previously missed matches in this evaluation.")

else:

    print(
        previously_missed[
            [
                "s1_id",
                "true_source_id",
                "current_hit",
                "new_hit",
                "current_candidate_count",
                "new_candidate_count",
            ]
        ].to_string(index=False)
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)