import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Original training data
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"

# Precomputed normalized data
NORMALIZED_DIR = (
    PROJECT_ROOT
    / "output"
    / "normalized"
)

# Token statistics
TOKEN_STATS_DIR = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "data"
    / "token_stats"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 1000
RANDOM_STATE = 42

BATCH_SIZE = 100_000
SINGLE_TOKEN_MAX_FREQ = 10_000


# ============================================================
# LOAD TOKEN STATISTICS
# ============================================================

print("Loading token statistics...")

with open(
    TOKEN_STATS_DIR / "name_token_counts.json",
    "r",
    encoding="utf-8"
) as f:
    name_counts = json.load(f)

with open(
    TOKEN_STATS_DIR / "address_token_counts.json",
    "r",
    encoding="utf-8"
) as f:
    address_counts = json.load(f)

print(f"Name tokens   : {len(name_counts):,}")
print(f"Address tokens: {len(address_counts):,}")


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("\nLoading Source 1...")

source1_path = (
    NORMALIZED_DIR
    / "train_source1_normalized.parquet"
)

source1 = pd.read_parquet(
    source1_path,
    columns=[
        "entity_id",
        "country",
        "norm_name",
        "compact_name",
        "norm_address"
    ]
)

sample_s1 = source1.sample(
    n=min(SAMPLE_SIZE, len(source1)),
    random_state=RANDOM_STATE
).reset_index(drop=True)

print(f"Source 1 total : {len(source1):,}")
print(f"Sample size    : {len(sample_s1):,}")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

ground_truth_path = (
    TRAIN_DIR
    / "train_ground_truth.parquet"
)

ground_truth = pd.read_parquet(
    ground_truth_path,
    columns=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

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

print(
    f"Ground-truth entries: "
    f"{len(gt_lookup):,}"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_tokens(normalized_value):
    """
    Return unique tokens with length >= 2.

    The input is already normalized, so no additional
    normalization is performed here.
    """

    if normalized_value is None:
        return set()

    value = str(normalized_value).strip()

    if not value:
        return set()

    return {
        token
        for token in value.split()
        if len(token) >= 2
    }


def rare_tokens(tokens, counts):
    """
    Select tokens whose global frequency is <= 10,000.
    """

    return [
        token
        for token in tokens
        if counts.get(
            token,
            10**18
        ) <= SINGLE_TOKEN_MAX_FREQ
    ]


def top_two_tokens(tokens, counts):
    """
    Select the two least-frequent tokens.
    """

    ranked = sorted(
        (
            counts.get(token, 10**18),
            token
        )
        for token in tokens
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

    return "|".join(sorted(tokens))


def clean_country(value):
    """
    Normalize country value for comparison.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).casefold().strip()


# ============================================================
# BUILD SOURCE 1 BLOCKING INDEXES
# ============================================================

print("\nBuilding Source 1 blocking indexes...")

query_info = {}

exact_name_lookup = defaultdict(set)
compact_name_lookup = defaultdict(set)

name_single_lookup = defaultdict(set)
name_pair_lookup = defaultdict(set)

address_single_lookup = defaultdict(set)
address_pair_lookup = defaultdict(set)


for row in sample_s1.itertuples(index=False):

    s1_id = row.entity_id

    country = clean_country(row.country)

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    normalized_name = (
        ""
        if row.norm_name is None
        else str(row.norm_name).strip()
    )

    compact_name = (
        ""
        if row.compact_name is None
        else str(row.compact_name).strip()
    )

    name_tokens = get_tokens(
        normalized_name
    )

    name_single = rare_tokens(
        name_tokens,
        name_counts
    )

    name_pair = make_pair(
        top_two_tokens(
            name_tokens,
            name_counts
        )
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    normalized_address = (
        ""
        if row.norm_address is None
        else str(row.norm_address).strip()
    )

    address_tokens = get_tokens(
        normalized_address
    )

    address_single = rare_tokens(
        address_tokens,
        address_counts
    )

    address_pair = make_pair(
        top_two_tokens(
            address_tokens,
            address_counts
        )
    )

    # --------------------------------------------------------
    # STORE COUNTRY
    # --------------------------------------------------------

    query_info[s1_id] = country

    # --------------------------------------------------------
    # EXACT NORMALIZED NAME
    # --------------------------------------------------------

    if normalized_name:

        exact_name_lookup[
            normalized_name
        ].add(s1_id)

    # --------------------------------------------------------
    # COMPACT NAME
    # --------------------------------------------------------

    if compact_name:

        compact_name_lookup[
            compact_name
        ].add(s1_id)

    # --------------------------------------------------------
    # RARE NAME TOKENS
    # --------------------------------------------------------

    for token in name_single:

        name_single_lookup[
            token
        ].add(s1_id)

    # --------------------------------------------------------
    # NAME TOKEN PAIR
    # --------------------------------------------------------

    if name_pair:

        name_pair_lookup[
            name_pair
        ].add(s1_id)

    # --------------------------------------------------------
    # RARE ADDRESS TOKENS
    # --------------------------------------------------------

    for token in address_single:

        address_single_lookup[
            token
        ].add(s1_id)

    # --------------------------------------------------------
    # ADDRESS TOKEN PAIR
    # --------------------------------------------------------

    if address_pair:

        address_pair_lookup[
            address_pair
        ].add(s1_id)


print(
    f"Exact name keys    : "
    f"{len(exact_name_lookup):,}"
)

print(
    f"Compact name keys  : "
    f"{len(compact_name_lookup):,}"
)

print(
    f"Name token keys    : "
    f"{len(name_single_lookup):,}"
)

print(
    f"Name pair keys     : "
    f"{len(name_pair_lookup):,}"
)

print(
    f"Address token keys : "
    f"{len(address_single_lookup):,}"
)

print(
    f"Address pair keys  : "
    f"{len(address_pair_lookup):,}"
)


# ============================================================
# CANDIDATE COUNTERS
# ============================================================

candidate_counts = defaultdict(int)

captured_matches = defaultdict(set)

rows_scanned = 0


# ============================================================
# SOURCE 2 + SOURCE 3
# ============================================================

source_files = [
    (
        NORMALIZED_DIR
        / "train_source2_normalized.parquet",
        "Source 2"
    ),
    (
        NORMALIZED_DIR
        / "train_source3_normalized.parquet",
        "Source 3"
    )
]


for source_path, source_label in source_files:

    print("\n" + "=" * 60)
    print(f"Scanning {source_label}")
    print("=" * 60)

    parquet_file = pq.ParquetFile(
        source_path
    )

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ),
        start=1
    ):

        chunk = batch.to_pandas()

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            country = clean_country(
                row.country
            )

            matched_s1 = set()

            # ------------------------------------------------
            # 1. EXACT NORMALIZED NAME
            # ------------------------------------------------

            normalized_name = (
                ""
                if row.norm_name is None
                else str(row.norm_name).strip()
            )

            if normalized_name:

                matched_s1.update(
                    exact_name_lookup.get(
                        normalized_name,
                        set()
                    )
                )

            # ------------------------------------------------
            # 2. COMPACT NAME
            # ------------------------------------------------

            compact_name = (
                ""
                if row.compact_name is None
                else str(row.compact_name).strip()
            )

            if compact_name:

                matched_s1.update(
                    compact_name_lookup.get(
                        compact_name,
                        set()
                    )
                )

            # ------------------------------------------------
            # 3. RARE NAME TOKENS
            # ------------------------------------------------

            name_tokens = get_tokens(
                normalized_name
            )

            candidate_name_single = rare_tokens(
                name_tokens,
                name_counts
            )

            for token in candidate_name_single:

                matched_s1.update(
                    name_single_lookup.get(
                        token,
                        set()
                    )
                )

            # ------------------------------------------------
            # 4. NAME TOKEN PAIR
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 5. RARE ADDRESS TOKENS
            # ------------------------------------------------

            address_value = (
                ""
                if row.norm_address is None
                else str(row.norm_address).strip()
            )

            address_tokens = get_tokens(
                address_value
            )

            candidate_address_single = rare_tokens(
                address_tokens,
                address_counts
            )

            for token in candidate_address_single:

                matched_s1.update(
                    address_single_lookup.get(
                        token,
                        set()
                    )
                )

            # ------------------------------------------------
            # 6. ADDRESS TOKEN PAIR
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 7. COUNTRY FILTER
            # ------------------------------------------------

            for s1_id in matched_s1:

                if country != query_info[s1_id]:
                    continue

                candidate_counts[s1_id] += 1

                # Check whether this candidate is actually
                # present in the ground truth.
                true_matches = gt_lookup.get(
                    s1_id,
                    set()
                )

                if entity_id in true_matches:

                    captured_matches[
                        s1_id
                    ].add(entity_id)

            rows_scanned += 1

        print(
            f"  Batch {batch_number:>3} "
            f"| rows scanned: {rows_scanned:,}"
        )

        del chunk


# ============================================================
# EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("BLOCKING SCALE BENCHMARK — 1,000 SOURCE 1 ENTITIES")
print("=" * 70)


results = []


for s1_id in sample_s1["entity_id"]:

    true_matches = gt_lookup.get(
        s1_id,
        set()
    )

    captured = captured_matches.get(
        s1_id,
        set()
    )

    candidate_count = candidate_counts.get(
        s1_id,
        0
    )

    if true_matches:

        recall = (
            len(captured)
            / len(true_matches)
        )

    else:

        recall = None

    results.append({
        "s1_id": s1_id,
        "true_matches": len(true_matches),
        "candidates": candidate_count,
        "captured": len(captured),
        "missed": len(true_matches - captured),
        "recall": recall
    })


results_df = pd.DataFrame(
    results
)

matched_entities = results_df[
    results_df["true_matches"] > 0
]


# ============================================================
# SUMMARY
# ============================================================

print(
    f"\nEntities evaluated       : "
    f"{len(results_df):,}"
)

print(
    f"Entities with true match : "
    f"{len(matched_entities):,}"
)

if not matched_entities.empty:

    print(
        f"100% recall entities     : "
        f"{(
            matched_entities["recall"] == 1.0
        ).sum():,}"
    )

    print(
        f"Entities with misses     : "
        f"{(
            matched_entities["missed"] > 0
        ).sum():,}"
    )

    print(
        f"Average blocking recall : "
        f"{matched_entities['recall'].mean():.4%}"
    )

else:

    print(
        "100% recall entities     : 0"
    )

    print(
        "Entities with misses     : 0"
    )

    print(
        "Average blocking recall : N/A"
    )


print(
    f"\nMean candidates          : "
    f"{results_df['candidates'].mean():,.2f}"
)

print(
    f"Median candidates        : "
    f"{results_df['candidates'].median():,.2f}"
)

print(
    f"95th percentile          : "
    f"{np.percentile(
        results_df["candidates"],
        95
    ):,.2f}"
)

print(
    f"Maximum candidates       : "
    f"{results_df['candidates'].max():,}"
)


# ============================================================
# ROUGH FULL-SCALE ESTIMATE
# ============================================================

mean_candidates = results_df[
    "candidates"
].mean()

estimated_pairs = (
    mean_candidates
    * len(source1)
)

print("\n" + "=" * 70)
print("ROUGH FULL-SCALE ESTIMATE")
print("=" * 70)

print(
    f"Mean candidates per S1  : "
    f"{mean_candidates:,.2f}"
)

print(
    f"Estimated candidate links "
    f"for all Source 1        : "
    f"{estimated_pairs:,.0f}"
)

print(
    "\nNOTE: This is only a rough estimate "
    "based on the 1,000-entity sample."
)


# ============================================================
# WORST RECALL CASES
# ============================================================

print("\n===== WORST RECALL CASES =====")

if not matched_entities.empty:

    print(
        matched_entities
        .sort_values(
            ["recall", "candidates"]
        )
        .head(10)
        .to_string(index=False)
    )

else:

    print("No entities with true matches found.")


# ============================================================
# LARGEST CANDIDATE SETS
# ============================================================

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