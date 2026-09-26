import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "train"
)

NORMALIZED_DIR = (
    PROJECT_ROOT
    / "output"
    / "normalized"
)

TOKEN_STATS_DIR = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "data"
    / "token_stats"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 1000
RANDOM_STATE = 42

BATCH_SIZE = 100_000

# Current V2 setting
SINGLE_TOKEN_MAX_FREQ = 10_000

# New strategy
MIN_SHARED_TOKENS = 2


# ============================================================
# FILE PATHS
# ============================================================

SOURCE1_PATH = (
    NORMALIZED_DIR
    / "train_source1_normalized.parquet"
)

SOURCE2_PATH = (
    NORMALIZED_DIR
    / "train_source2_normalized.parquet"
)

SOURCE3_PATH = (
    NORMALIZED_DIR
    / "train_source3_normalized.parquet"
)

GROUND_TRUTH_PATH = (
    TRAIN_DIR
    / "train_ground_truth.parquet"
)

NAME_STATS_PATH = (
    TOKEN_STATS_DIR
    / "name_token_counts.json"
)

ADDRESS_STATS_PATH = (
    TOKEN_STATS_DIR
    / "address_token_counts.json"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_text(value):
    """
    Safely convert a value into a stripped string.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_country(value):
    """
    Normalize country for comparison.
    """

    return safe_text(value).casefold()


def get_tokens(normalized_value):
    """
    Return unique normalized tokens.

    One-character tokens are ignored.
    """

    value = safe_text(normalized_value)

    if not value:
        return set()

    return {
        token
        for token in value.split()
        if len(token) >= 2
    }


def rare_tokens(
    tokens,
    counts,
    max_frequency
):
    """
    Return tokens whose global frequency is
    below or equal to max_frequency.
    """

    return [
        token
        for token in tokens
        if counts.get(
            token,
            10**18
        ) <= max_frequency
    ]


def rarest_n_tokens(
    tokens,
    counts,
    n
):
    """
    Return the n least-frequent tokens.

    Frequency is taken from the global token statistics.
    """

    ranked = sorted(
        (
            counts.get(
                token,
                10**18
            ),
            token
        )
        for token in tokens
    )

    return [
        token
        for _, token in ranked[:n]
    ]


def make_pair_key(tokens):
    """
    Create an order-independent pair key.
    """

    if len(tokens) < 2:
        return None

    return "|".join(
        sorted(tokens)
    )


# ============================================================
# LOAD TOKEN STATISTICS
# ============================================================

print("=" * 80)
print("LOADING TOKEN STATISTICS")
print("=" * 80)

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
    f"Name tokens   : {len(name_counts):,}"
)

print(
    f"Address tokens: {len(address_counts):,}"
)


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("\n" + "=" * 80)
print("LOADING SOURCE 1")
print("=" * 80)

source1 = pd.read_parquet(
    SOURCE1_PATH,
    columns=[
        "entity_id",
        "country",
        "norm_name",
        "compact_name",
        "norm_address"
    ]
)

print(
    f"Source 1 rows: {len(source1):,}"
)


# ============================================================
# SAMPLE SOURCE 1
# ============================================================

s1_sample = source1.sample(
    n=min(
        SAMPLE_SIZE,
        len(source1)
    ),
    random_state=RANDOM_STATE
).reset_index(
    drop=True
)

print(
    f"Sample size : {len(s1_sample):,}"
)

print(
    f"Random state: {RANDOM_STATE}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\n" + "=" * 80)
print("LOADING GROUND TRUTH")
print("=" * 80)

ground_truth = pd.read_parquet(
    GROUND_TRUTH_PATH,
    columns=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

print(
    f"Ground truth rows: {len(ground_truth):,}"
)


# ============================================================
# BUILD GROUND-TRUTH LOOKUP
# ============================================================

gt_lookup = {}

sample_ids = set(
    s1_sample["entity_id"]
)


for row in ground_truth.itertuples(
    index=False
):

    s1_id = row.source1_entity_id

    # We only need ground truth for our
    # 1,000 sampled Source 1 entities.
    if s1_id not in sample_ids:
        continue

    value = row.matched_entity_ids

    if pd.isna(value):

        matches = set()

    elif str(value).strip() == "":

        matches = set()

    else:

        matches = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    gt_lookup[s1_id] = matches


print(
    f"Sample ground-truth entries: "
    f"{len(gt_lookup):,}"
)


# ============================================================
# BUILD QUERY-SIDE INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("BUILDING SOURCE 1 BLOCKING INDEXES")
print("=" * 80)


query_country = {}

query_name_tokens = {}
query_address_tokens = {}


# ------------------------------------------------------------
# CURRENT V2 INDEXES
# ------------------------------------------------------------

exact_name_lookup = defaultdict(set)

compact_name_lookup = defaultdict(set)

name_single_lookup = defaultdict(set)

name_pair_lookup = defaultdict(set)

address_single_lookup = defaultdict(set)

address_pair_lookup = defaultdict(set)


# ------------------------------------------------------------
# NEW TWO-SIGNAL INDEXES
#
# IMPORTANT:
# These indexes contain all normalized tokens.
# We do NOT apply the old 10,000-frequency limit here.
#
# The new strategy requires at least TWO shared tokens,
# which provides the selectivity.
# ------------------------------------------------------------

all_name_token_lookup = defaultdict(set)

all_address_token_lookup = defaultdict(set)


# ============================================================
# BUILD INDEXES
# ============================================================

for row in s1_sample.itertuples(
    index=False
):

    s1_id = row.entity_id

    country = clean_country(
        row.country
    )

    name = safe_text(
        row.norm_name
    )

    compact_name = safe_text(
        row.compact_name
    )

    address = safe_text(
        row.norm_address
    )

    name_tokens = get_tokens(
        name
    )

    address_tokens = get_tokens(
        address
    )

    query_country[s1_id] = country

    query_name_tokens[s1_id] = name_tokens

    query_address_tokens[s1_id] = address_tokens


    # ========================================================
    # CURRENT V2
    # ========================================================

    # Exact normalized name

    if name:

        exact_name_lookup[
            name
        ].add(s1_id)


    # Compact name

    if compact_name:

        compact_name_lookup[
            compact_name
        ].add(s1_id)


    # Rare name tokens

    rare_name = rare_tokens(
        name_tokens,
        name_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    for token in rare_name:

        name_single_lookup[
            token
        ].add(s1_id)


    # Name token pair

    name_pair = make_pair_key(
        rarest_n_tokens(
            name_tokens,
            name_counts,
            2
        )
    )

    if name_pair:

        name_pair_lookup[
            name_pair
        ].add(s1_id)


    # Rare address tokens

    rare_address = rare_tokens(
        address_tokens,
        address_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    for token in rare_address:

        address_single_lookup[
            token
        ].add(s1_id)


    # Address token pair

    address_pair = make_pair_key(
        rarest_n_tokens(
            address_tokens,
            address_counts,
            2
        )
    )

    if address_pair:

        address_pair_lookup[
            address_pair
        ].add(s1_id)


    # ========================================================
    # NEW TWO-SIGNAL INDEXES
    # ========================================================

    for token in name_tokens:

        all_name_token_lookup[
            token
        ].add(s1_id)


    for token in address_tokens:

        all_address_token_lookup[
            token
        ].add(s1_id)


# ============================================================
# INDEX SUMMARY
# ============================================================

print(
    f"Exact name keys    : "
    f"{len(exact_name_lookup):,}"
)

print(
    f"Compact name keys  : "
    f"{len(compact_name_lookup):,}"
)

print(
    f"Rare name keys     : "
    f"{len(name_single_lookup):,}"
)

print(
    f"Name pair keys     : "
    f"{len(name_pair_lookup):,}"
)

print(
    f"Rare address keys  : "
    f"{len(address_single_lookup):,}"
)

print(
    f"Address pair keys  : "
    f"{len(address_pair_lookup):,}"
)

print(
    f"All name token keys: "
    f"{len(all_name_token_lookup):,}"
)

print(
    f"All address tokens : "
    f"{len(all_address_token_lookup):,}"
)


# ============================================================
# CANDIDATE COUNTERS
# ============================================================

current_candidate_counts = defaultdict(int)

new_candidate_counts = defaultdict(int)


# ============================================================
# TRUE MATCH CAPTURE
#
# We only store captured TRUE matches.
# This is tiny compared with storing every candidate pair.
# ============================================================

current_captured = defaultdict(set)

new_captured = defaultdict(set)


# ============================================================
# SOURCE SCANNING
# ============================================================

source_files = [
    (
        SOURCE2_PATH,
        "Source 2"
    ),
    (
        SOURCE3_PATH,
        "Source 3"
    )
]


total_rows_scanned = 0

overall_start = time.time()


for source_path, source_label in source_files:

    print("\n" + "=" * 80)
    print(f"SCANNING {source_label}")
    print("=" * 80)

    source_start = time.time()

    parquet_file = pq.ParquetFile(
        source_path
    )

    rows_in_source = 0

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=[
                "entity_id",
                "country",
                "norm_name",
                "compact_name",
                "norm_address"
            ]
        ),
        start=1
    ):

        chunk = batch.to_pandas()

        # ----------------------------------------------------
        # PROCESS CURRENT BATCH
        # ----------------------------------------------------

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = row.entity_id

            country = clean_country(
                row.country
            )

            name = safe_text(
                row.norm_name
            )

            compact_name = safe_text(
                row.compact_name
            )

            address = safe_text(
                row.norm_address
            )

            name_tokens = get_tokens(
                name
            )

            address_tokens = get_tokens(
                address
            )


            # =================================================
            # CURRENT V2 STRATEGY
            # =================================================

            current_matched = set()


            # -------------------------------------------------
            # 1. Exact normalized name
            # -------------------------------------------------

            if name:

                current_matched.update(
                    exact_name_lookup.get(
                        name,
                        ()
                    )
                )


            # -------------------------------------------------
            # 2. Compact name
            # -------------------------------------------------

            if compact_name:

                current_matched.update(
                    compact_name_lookup.get(
                        compact_name,
                        ()
                    )
                )


            # -------------------------------------------------
            # 3. Rare name tokens
            # -------------------------------------------------

            rare_name = rare_tokens(
                name_tokens,
                name_counts,
                SINGLE_TOKEN_MAX_FREQ
            )

            for token in rare_name:

                current_matched.update(
                    name_single_lookup.get(
                        token,
                        ()
                    )
                )


            # -------------------------------------------------
            # 4. Name pair
            # -------------------------------------------------

            name_pair = make_pair_key(
                rarest_n_tokens(
                    name_tokens,
                    name_counts,
                    2
                )
            )

            if name_pair:

                current_matched.update(
                    name_pair_lookup.get(
                        name_pair,
                        ()
                    )
                )


            # -------------------------------------------------
            # 5. Rare address tokens
            # -------------------------------------------------

            rare_address = rare_tokens(
                address_tokens,
                address_counts,
                SINGLE_TOKEN_MAX_FREQ
            )

            for token in rare_address:

                current_matched.update(
                    address_single_lookup.get(
                        token,
                        ()
                    )
                )


            # -------------------------------------------------
            # 6. Address pair
            # -------------------------------------------------

            address_pair = make_pair_key(
                rarest_n_tokens(
                    address_tokens,
                    address_counts,
                    2
                )
            )

            if address_pair:

                current_matched.update(
                    address_pair_lookup.get(
                        address_pair,
                        ()
                    )
                )


            # -------------------------------------------------
            # Country filter
            # -------------------------------------------------

            current_matched = {
                s1_id
                for s1_id in current_matched
                if country == query_country[s1_id]
            }


            # =================================================
            # NEW TWO-SIGNAL STRATEGY
            # =================================================

            # We count how many tokens from the current
            # Source 2/3 record are shared with each sampled
            # Source 1 record.

            name_overlap_counts = defaultdict(int)

            address_overlap_counts = defaultdict(int)


            # -------------------------------------------------
            # Name token overlap
            # -------------------------------------------------

            for token in name_tokens:

                for s1_id in all_name_token_lookup.get(
                    token,
                    ()
                ):

                    name_overlap_counts[
                        s1_id
                    ] += 1


            # -------------------------------------------------
            # Address token overlap
            # -------------------------------------------------

            for token in address_tokens:

                for s1_id in all_address_token_lookup.get(
                    token,
                    ()
                ):

                    address_overlap_counts[
                        s1_id
                    ] += 1


            # -------------------------------------------------
            # Select candidates with >= 2 shared name tokens
            # -------------------------------------------------

            new_matched = set()

            for s1_id, shared_count in name_overlap_counts.items():

                if shared_count >= MIN_SHARED_TOKENS:

                    new_matched.add(
                        s1_id
                    )


            # -------------------------------------------------
            # Select candidates with >= 2 shared address tokens
            # -------------------------------------------------

            for s1_id, shared_count in address_overlap_counts.items():

                if shared_count >= MIN_SHARED_TOKENS:

                    new_matched.add(
                        s1_id
                    )


            # -------------------------------------------------
            # Strong exact-name rules remain in new strategy
            # -------------------------------------------------

            if name:

                new_matched.update(
                    exact_name_lookup.get(
                        name,
                        ()
                    )
                )


            if compact_name:

                new_matched.update(
                    compact_name_lookup.get(
                        compact_name,
                        ()
                    )
                )


            # -------------------------------------------------
            # Country filter
            # -------------------------------------------------

            new_matched = {
                s1_id
                for s1_id in new_matched
                if country == query_country[s1_id]
            }


            # =================================================
            # UPDATE CANDIDATE COUNTS
            # =================================================

            for s1_id in current_matched:

                current_candidate_counts[
                    s1_id
                ] += 1


            for s1_id in new_matched:

                new_candidate_counts[
                    s1_id
                ] += 1


            # =================================================
            # CHECK TRUE MATCHES
            # =================================================

            for s1_id in current_matched:

                true_matches = gt_lookup.get(
                    s1_id,
                    set()
                )

                if entity_id in true_matches:

                    current_captured[
                        s1_id
                    ].add(
                        entity_id
                    )


            for s1_id in new_matched:

                true_matches = gt_lookup.get(
                    s1_id,
                    set()
                )

                if entity_id in true_matches:

                    new_captured[
                        s1_id
                    ].add(
                        entity_id
                    )


            rows_in_source += 1
            total_rows_scanned += 1


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        print(
            f"  Batch {batch_number:>3}"
            f" | rows scanned: {rows_in_source:,}"
        )

        del chunk


    print(
        f"{source_label} completed in "
        f"{time.time() - source_start:.1f}s"
    )


# ============================================================
# BUILD RESULTS
# ============================================================

print("\n" + "=" * 80)
print("BUILDING BENCHMARK RESULTS")
print("=" * 80)


results = []


for s1_id in s1_sample["entity_id"]:

    true_matches = gt_lookup.get(
        s1_id,
        set()
    )


    current_candidates = current_candidate_counts.get(
        s1_id,
        0
    )

    new_candidates = new_candidate_counts.get(
        s1_id,
        0
    )


    current_match_set = current_captured.get(
        s1_id,
        set()
    )

    new_match_set = new_captured.get(
        s1_id,
        set()
    )


    if true_matches:

        current_recall = (
            len(current_match_set)
            / len(true_matches)
        )

        new_recall = (
            len(new_match_set)
            / len(true_matches)
        )

    else:

        current_recall = None
        new_recall = None


    recovered = (
        true_matches
        - current_match_set
    ) & new_match_set


    regressions = (
        current_match_set
        - new_match_set
    )


    results.append({
        "s1_id": s1_id,
        "true_matches": len(true_matches),
        "current_candidates": current_candidates,
        "new_candidates": new_candidates,
        "current_captured": len(current_match_set),
        "new_captured": len(new_match_set),
        "current_recall": current_recall,
        "new_recall": new_recall,
        "recovered_missed_matches": len(recovered),
        "regressions": len(regressions)
    })


results_df = pd.DataFrame(
    results
)


scored = results_df[
    results_df["true_matches"] > 0
].copy()


# ============================================================
# MAIN SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("TWO-SIGNAL BLOCKING BENCHMARK")
print("=" * 80)

print(
    f"\nEntities evaluated       : "
    f"{len(results_df):,}"
)

print(
    f"Entities with true match : "
    f"{len(scored):,}"
)


# ============================================================
# RECALL
# ============================================================

print("\n----- BLOCKING RECALL -----")

if not scored.empty:

    print(
        f"Current V2 recall       : "
        f"{scored['current_recall'].mean():.4%}"
    )

    print(
        f"New two-signal recall   : "
        f"{scored['new_recall'].mean():.4%}"
    )

    print(
        f"Current 100% recall     : "
        f"{(
            scored["current_recall"] == 1.0
        ).sum():,}"
    )

    print(
        f"New 100% recall         : "
        f"{(
            scored["new_recall"] == 1.0
        ).sum():,}"
    )

else:

    print(
        "No entities with true matches."
    )


# ============================================================
# CANDIDATE SIZE
# ============================================================

print("\n----- CANDIDATE SET SIZE -----")

print(
    f"Current mean            : "
    f"{results_df['current_candidates'].mean():,.2f}"
)

print(
    f"New mean                : "
    f"{results_df['new_candidates'].mean():,.2f}"
)

print(
    f"Current median          : "
    f"{results_df['current_candidates'].median():,.2f}"
)

print(
    f"New median              : "
    f"{results_df['new_candidates'].median():,.2f}"
)

print(
    f"Current 95th percentile : "
    f"{np.percentile(
        results_df['current_candidates'],
        95
    ):,.2f}"
)

print(
    f"New 95th percentile     : "
    f"{np.percentile(
        results_df['new_candidates'],
        95
    ):,.2f}"
)

print(
    f"Current maximum         : "
    f"{results_df['current_candidates'].max():,}"
)

print(
    f"New maximum             : "
    f"{results_df['new_candidates'].max():,}"
)


# ============================================================
# RECOVERY / REGRESSION
# ============================================================

total_recovered = int(
    scored[
        "recovered_missed_matches"
    ].sum()
)

total_regressions = int(
    scored[
        "regressions"
    ].sum()
)


print("\n----- MATCH RECOVERY -----")

print(
    f"Missed matches recovered by NEW : "
    f"{total_recovered}"
)

print(
    f"Entities with recovery           : "
    f"{(
        scored['recovered_missed_matches'] > 0
    ).sum()}"
)

print(
    f"Current matches lost by NEW      : "
    f"{total_regressions}"
)


# ============================================================
# FULL-SCALE ESTIMATE
# ============================================================

current_mean = results_df[
    "current_candidates"
].mean()

new_mean = results_df[
    "new_candidates"
].mean()


estimated_current = (
    current_mean
    * len(source1)
)

estimated_new = (
    new_mean
    * len(source1)
)


print("\n" + "=" * 80)
print("ROUGH FULL-SCALE ESTIMATE")
print("=" * 80)

print(
    f"Source 1 entities      : "
    f"{len(source1):,}"
)

print(
    f"Current mean/S1        : "
    f"{current_mean:,.2f}"
)

print(
    f"New mean/S1            : "
    f"{new_mean:,.2f}"
)

print(
    f"\nCurrent estimated pairs: "
    f"{estimated_current:,.0f}"
)

print(
    f"New estimated pairs    : "
    f"{estimated_new:,.0f}"
)


if current_mean > 0:

    reduction = (
        1
        - (
            new_mean
            / current_mean
        )
    ) * 100

    print(
        f"\nEstimated candidate "
        f"reduction             : "
        f"{reduction:.2f}%"
    )


# ============================================================
# WORST RECALL CASES
# ============================================================

print("\n" + "=" * 80)
print("WORST RECALL CASES — NEW STRATEGY")
print("=" * 80)


if not scored.empty:

    print(
        scored
        .sort_values(
            [
                "new_recall",
                "new_candidates"
            ]
        )
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# BIGGEST CANDIDATE SETS
# ============================================================

print("\n" + "=" * 80)
print("LARGEST NEW CANDIDATE SETS")
print("=" * 80)

print(
    results_df
    .sort_values(
        "new_candidates",
        ascending=False
    )
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_path = (
    OUTPUT_DIR
    / "benchmark_two_signal_1000_results.csv"
)

results_df.to_csv(
    result_path,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)

print(
    f"Total rows scanned: "
    f"{total_rows_scanned:,}"
)

print(
    f"Total runtime: "
    f"{time.time() - overall_start:.1f}s"
)

print(
    f"Results saved to:"
)

print(
    result_path
)