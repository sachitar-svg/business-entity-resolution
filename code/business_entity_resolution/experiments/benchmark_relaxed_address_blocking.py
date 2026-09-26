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
# FILES
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

BASELINE_RESULTS_PATH = (
    OUTPUT_DIR
    / "benchmark_two_signal_1000_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 1000
RANDOM_STATE = 42

BATCH_SIZE = 100_000

CURRENT_MAX_FREQ = 10_000

RELAXED_25K_MAX_FREQ = 25_000

RELAXED_50K_MAX_FREQ = 50_000


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):
    """
    Safely convert a value to a stripped string.
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


def get_tokens(value):
    """
    Get unique normalized tokens with length >= 2.

    Input is already normalized.
    """

    value = safe_text(value)

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
    Keep tokens whose global frequency is <= max_frequency.
    """

    return {
        token
        for token in tokens
        if counts.get(
            token,
            10**18
        ) <= max_frequency
    }


def rarest_two_tokens(
    tokens,
    counts
):
    """
    Return the two least-frequent tokens.
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
# LOAD SOURCE 1 SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("LOADING SOURCE 1 SAMPLE")
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

s1_sample = source1.sample(
    n=min(
        SAMPLE_SIZE,
        len(source1)
    ),
    random_state=RANDOM_STATE
).reset_index(
    drop=True
)

sample_ids = list(
    s1_sample["entity_id"]
)

sample_id_to_index = {
    entity_id: index
    for index, entity_id
    in enumerate(sample_ids)
}


print(
    f"Source 1 total : {len(source1):,}"
)

print(
    f"Sample size    : {len(s1_sample):,}"
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


# One ground-truth set per sampled Source 1 index

gt_by_index = [
    set()
    for _ in range(len(s1_sample))
]


for row in ground_truth.itertuples(
    index=False
):

    s1_index = sample_id_to_index.get(
        row.source1_entity_id
    )

    if s1_index is None:
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

    gt_by_index[
        s1_index
    ] = matches


total_true_pairs = sum(
    len(matches)
    for matches in gt_by_index
)


entities_with_matches = sum(
    bool(matches)
    for matches in gt_by_index
)


print(
    f"Entities with true matches: "
    f"{entities_with_matches:,}"
)

print(
    f"True pairs in sample: "
    f"{total_true_pairs:,}"
)


# ============================================================
# LOAD BASELINE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("LOADING CURRENT V2 BASELINE RESULTS")
print("=" * 80)

if not BASELINE_RESULTS_PATH.exists():

    raise FileNotFoundError(
        "Expected baseline results file was not found:\n"
        f"{BASELINE_RESULTS_PATH}\n\n"
        "Please make sure benchmark_two_signal_blocking.py "
        "has already been run."
    )


baseline_df = pd.read_csv(
    BASELINE_RESULTS_PATH
)


print(
    f"Baseline rows: {len(baseline_df):,}"
)


baseline_mean = (
    baseline_df[
        "current_candidates"
    ].mean()
)

baseline_median = (
    baseline_df[
        "current_candidates"
    ].median()
)

baseline_95 = np.percentile(
    baseline_df[
        "current_candidates"
    ],
    95
)

baseline_max = (
    baseline_df[
        "current_candidates"
    ].max()
)


print(
    f"Baseline mean candidates: "
    f"{baseline_mean:,.2f}"
)

print(
    f"Baseline median: "
    f"{baseline_median:,.2f}"
)

print(
    f"Baseline maximum: "
    f"{baseline_max:,}"
)


# ============================================================
# BUILD SOURCE 1 BLOCKING INDEXES
# ============================================================

print("\n" + "=" * 80)
print("BUILDING SOURCE 1 BLOCKING INDEXES")
print("=" * 80)


query_country = [
    ""
    for _ in range(
        len(s1_sample)
    )
]


# ------------------------------------------------------------
# Current V2 indexes
# ------------------------------------------------------------

exact_name_lookup = defaultdict(list)

compact_name_lookup = defaultdict(list)

name_single_lookup = defaultdict(list)

name_pair_lookup = defaultdict(list)

address_single_lookup = defaultdict(list)

address_pair_lookup = defaultdict(list)


# ------------------------------------------------------------
# Relaxed address indexes
# ------------------------------------------------------------

address_relaxed_25k_lookup = defaultdict(list)

address_relaxed_50k_lookup = defaultdict(list)


# ============================================================
# BUILD
# ============================================================

for s1_index, row in enumerate(
    s1_sample.itertuples(
        index=False
    )
):

    query_country[
        s1_index
    ] = clean_country(
        row.country
    )


    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    name = safe_text(
        row.norm_name
    )

    compact_name = safe_text(
        row.compact_name
    )

    name_tokens = get_tokens(
        name
    )


    # Exact name

    if name:

        exact_name_lookup[
            name
        ].append(
            s1_index
        )


    # Compact name

    if compact_name:

        compact_name_lookup[
            compact_name
        ].append(
            s1_index
        )


    # Rare name tokens

    rare_name = rare_tokens(
        name_tokens,
        name_counts,
        CURRENT_MAX_FREQ
    )

    for token in rare_name:

        name_single_lookup[
            token
        ].append(
            s1_index
        )


    # Name pair

    name_pair = make_pair(
        rarest_two_tokens(
            name_tokens,
            name_counts
        )
    )

    if name_pair:

        name_pair_lookup[
            name_pair
        ].append(
            s1_index
        )


    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------

    address = safe_text(
        row.norm_address
    )

    address_tokens = get_tokens(
        address
    )


    # Rare address tokens <= 10k

    rare_address = rare_tokens(
        address_tokens,
        address_counts,
        CURRENT_MAX_FREQ
    )

    for token in rare_address:

        address_single_lookup[
            token
        ].append(
            s1_index
        )


    # Address pair

    address_pair = make_pair(
        rarest_two_tokens(
            address_tokens,
            address_counts
        )
    )

    if address_pair:

        address_pair_lookup[
            address_pair
        ].append(
            s1_index
        )


    # --------------------------------------------------------
    # Relaxed address: 10,001 - 25,000
    # --------------------------------------------------------

    for token in address_tokens:

        frequency = address_counts.get(
            token,
            10**18
        )

        if (
            CURRENT_MAX_FREQ
            < frequency
            <= RELAXED_25K_MAX_FREQ
        ):

            address_relaxed_25k_lookup[
                token
            ].append(
                s1_index
            )


    # --------------------------------------------------------
    # Relaxed address: 10,001 - 50,000
    # --------------------------------------------------------

    for token in address_tokens:

        frequency = address_counts.get(
            token,
            10**18
        )

        if (
            CURRENT_MAX_FREQ
            < frequency
            <= RELAXED_50K_MAX_FREQ
        ):

            address_relaxed_50k_lookup[
                token
            ].append(
                s1_index
            )


print(
    f"Exact name keys            : "
    f"{len(exact_name_lookup):,}"
)

print(
    f"Compact name keys          : "
    f"{len(compact_name_lookup):,}"
)

print(
    f"Rare name keys             : "
    f"{len(name_single_lookup):,}"
)

print(
    f"Name pair keys             : "
    f"{len(name_pair_lookup):,}"
)

print(
    f"Rare address keys          : "
    f"{len(address_single_lookup):,}"
)

print(
    f"Address pair keys          : "
    f"{len(address_pair_lookup):,}"
)

print(
    f"Relaxed address 25k keys   : "
    f"{len(address_relaxed_25k_lookup):,}"
)

print(
    f"Relaxed address 50k keys   : "
    f"{len(address_relaxed_50k_lookup):,}"
)


# ============================================================
# TRUE MATCH CAPTURE
# ============================================================

current_captured = [
    set()
    for _ in range(
        len(s1_sample)
    )
]

relaxed_25k_captured = [
    set()
    for _ in range(
        len(s1_sample)
    )
]

relaxed_50k_captured = [
    set()
    for _ in range(
        len(s1_sample)
    )
]


# ============================================================
# CANDIDATE SET STORAGE
#
# We store a compact integer row ID rather than the full
# entity_id string to reduce memory usage.
#
# 25k strategy = Current V2 + 10,001-25,000 address tokens
# 50k strategy = Current V2 + 10,001-50,000 address tokens
# ============================================================

candidate_sets_25k = [
    set()
    for _ in range(
        len(s1_sample)
    )
]

candidate_sets_50k = [
    set()
    for _ in range(
        len(s1_sample)
    )
]


# ============================================================
# SCAN SOURCE 2 + SOURCE 3
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


global_row_id = 0

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


            # =================================================
            # CURRENT V2 MATCHES
            # =================================================

            current_matches = set()


            # -------------------------------------------------
            # 1. Exact normalized name
            # -------------------------------------------------

            if name:

                current_matches.update(
                    exact_name_lookup.get(
                        name,
                        ()
                    )
                )


            # -------------------------------------------------
            # 2. Compact name
            # -------------------------------------------------

            if compact_name:

                current_matches.update(
                    compact_name_lookup.get(
                        compact_name,
                        ()
                    )
                )


            # -------------------------------------------------
            # 3. Rare name tokens
            # -------------------------------------------------

            name_tokens = get_tokens(
                name
            )

            rare_name = rare_tokens(
                name_tokens,
                name_counts,
                CURRENT_MAX_FREQ
            )

            for token in rare_name:

                current_matches.update(
                    name_single_lookup.get(
                        token,
                        ()
                    )
                )


            # -------------------------------------------------
            # 4. Name pair
            # -------------------------------------------------

            name_pair = make_pair(
                rarest_two_tokens(
                    name_tokens,
                    name_counts
                )
            )

            if name_pair:

                current_matches.update(
                    name_pair_lookup.get(
                        name_pair,
                        ()
                    )
                )


            # -------------------------------------------------
            # 5. Rare address tokens <= 10k
            # -------------------------------------------------

            address_tokens = get_tokens(
                address
            )

            rare_address = rare_tokens(
                address_tokens,
                address_counts,
                CURRENT_MAX_FREQ
            )

            for token in rare_address:

                current_matches.update(
                    address_single_lookup.get(
                        token,
                        ()
                    )
                )


            # -------------------------------------------------
            # 6. Address pair
            # -------------------------------------------------

            address_pair = make_pair(
                rarest_two_tokens(
                    address_tokens,
                    address_counts
                )
            )

            if address_pair:

                current_matches.update(
                    address_pair_lookup.get(
                        address_pair,
                        ()
                    )
                )


            # -------------------------------------------------
            # Country filter
            # -------------------------------------------------

            current_matches = {
                s1_index
                for s1_index
                in current_matches
                if country
                == query_country[s1_index]
            }


            # =================================================
            # RELAXED ADDRESS 25K
            # =================================================

            relaxed_25k_matches = set(
                current_matches
            )


            for token in address_tokens:

                frequency = address_counts.get(
                    token,
                    10**18
                )

                if (
                    CURRENT_MAX_FREQ
                    < frequency
                    <= RELAXED_25K_MAX_FREQ
                ):

                    relaxed_25k_matches.update(
                        address_relaxed_25k_lookup.get(
                            token,
                            ()
                        )
                    )


            # Country filter for new matches

            relaxed_25k_matches = {
                s1_index
                for s1_index
                in relaxed_25k_matches
                if country
                == query_country[s1_index]
            }


            # =================================================
            # RELAXED ADDRESS 50K
            # =================================================

            relaxed_50k_matches = set(
                current_matches
            )


            for token in address_tokens:

                frequency = address_counts.get(
                    token,
                    10**18
                )

                if (
                    CURRENT_MAX_FREQ
                    < frequency
                    <= RELAXED_50K_MAX_FREQ
                ):

                    relaxed_50k_matches.update(
                        address_relaxed_50k_lookup.get(
                            token,
                            ()
                        )
                    )


            relaxed_50k_matches = {
                s1_index
                for s1_index
                in relaxed_50k_matches
                if country
                == query_country[s1_index]
            }


            # =================================================
            # STORE CANDIDATE ROW IDS
            # =================================================

            for s1_index in relaxed_25k_matches:

                candidate_sets_25k[
                    s1_index
                ].add(
                    global_row_id
                )


            for s1_index in relaxed_50k_matches:

                candidate_sets_50k[
                    s1_index
                ].add(
                    global_row_id
                )


            # =================================================
            # TRUE MATCH CAPTURE
            # =================================================

            for s1_index in current_matches:

                if (
                    entity_id
                    in gt_by_index[s1_index]
                ):

                    current_captured[
                        s1_index
                    ].add(
                        entity_id
                    )


            for s1_index in relaxed_25k_matches:

                if (
                    entity_id
                    in gt_by_index[s1_index]
                ):

                    relaxed_25k_captured[
                        s1_index
                    ].add(
                        entity_id
                    )


            for s1_index in relaxed_50k_matches:

                if (
                    entity_id
                    in gt_by_index[s1_index]
                ):

                    relaxed_50k_captured[
                        s1_index
                    ].add(
                        entity_id
                    )


            global_row_id += 1

            total_rows_scanned += 1


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        print(
            f"  Batch {batch_number:>3}"
            f" | rows scanned: "
            f"{total_rows_scanned:,}"
        )

        del chunk


    print(
        f"{source_label} completed in "
        f"{time.time() - source_start:.1f}s"
    )


# ============================================================
# BUILD RESULT DATAFRAME
# ============================================================

print("\n" + "=" * 80)
print("BUILDING RESULTS")
print("=" * 80)


results = []


for s1_index, s1_id in enumerate(
    sample_ids
):

    true_matches = gt_by_index[
        s1_index
    ]


    baseline_row = baseline_df[
        baseline_df["s1_id"] == s1_id
    ]


    if baseline_row.empty:

        baseline_candidates = np.nan

    else:

        baseline_candidates = (
            baseline_row[
                "current_candidates"
            ].iloc[0]
        )


    current_captured_count = len(
        current_captured[
            s1_index
        ]
    )

    relaxed_25k_captured_count = len(
        relaxed_25k_captured[
            s1_index
        ]
    )

    relaxed_50k_captured_count = len(
        relaxed_50k_captured[
            s1_index
        ]
    )


    if true_matches:

        current_recall = (
            current_captured_count
            / len(true_matches)
        )

        recall_25k = (
            relaxed_25k_captured_count
            / len(true_matches)
        )

        recall_50k = (
            relaxed_50k_captured_count
            / len(true_matches)
        )

    else:

        current_recall = np.nan
        recall_25k = np.nan
        recall_50k = np.nan


    results.append({

        "s1_id": s1_id,

        "true_matches": len(
            true_matches
        ),

        "baseline_candidates": (
            baseline_candidates
        ),

        "current_captured": (
            current_captured_count
        ),

        "current_recall": (
            current_recall
        ),

        "candidates_25k": len(
            candidate_sets_25k[
                s1_index
            ]
        ),

        "captured_25k": (
            relaxed_25k_captured_count
        ),

        "recall_25k": recall_25k,

        "candidates_50k": len(
            candidate_sets_50k[
                s1_index
            ]
        ),

        "captured_50k": (
            relaxed_50k_captured_count
        ),

        "recall_50k": recall_50k
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("RELAXED ADDRESS BLOCKING BENCHMARK")
print("=" * 80)


print(
    f"\nEntities evaluated       : "
    f"{len(results_df):,}"
)

print(
    f"Entities with true match : "
    f"{entities_with_matches:,}"
)


# ============================================================
# RECALL
# ============================================================

print("\n----- BLOCKING RECALL -----")

scored = results_df[
    results_df["true_matches"] > 0
]


print(
    f"Current V2 recall        : "
    f"{scored['current_recall'].mean():.4%}"
)

print(
    f"V2 + address 25k recall  : "
    f"{scored['recall_25k'].mean():.4%}"
)

print(
    f"V2 + address 50k recall  : "
    f"{scored['recall_50k'].mean():.4%}"
)


# ============================================================
# CANDIDATE SIZE
# ============================================================

print("\n----- CANDIDATE SET SIZE -----")


print(
    f"Current V2 mean          : "
    f"{results_df['baseline_candidates'].mean():,.2f}"
)

print(
    f"25k mean                 : "
    f"{results_df['candidates_25k'].mean():,.2f}"
)

print(
    f"50k mean                 : "
    f"{results_df['candidates_50k'].mean():,.2f}"
)


print(
    f"Current V2 median        : "
    f"{results_df['baseline_candidates'].median():,.2f}"
)

print(
    f"25k median               : "
    f"{results_df['candidates_25k'].median():,.2f}"
)

print(
    f"50k median               : "
    f"{results_df['candidates_50k'].median():,.2f}"
)


print(
    f"\nCurrent V2 95th pct      : "
    f"{np.percentile(
        results_df['baseline_candidates'].dropna(),
        95
    ):,.2f}"
)

print(
    f"25k 95th pct             : "
    f"{np.percentile(
        results_df['candidates_25k'],
        95
    ):,.2f}"
)

print(
    f"50k 95th pct             : "
    f"{np.percentile(
        results_df['candidates_50k'],
        95
    ):,.2f}"
)


print(
    f"\nCurrent V2 maximum       : "
    f"{results_df['baseline_candidates'].max():,.0f}"
)

print(
    f"25k maximum              : "
    f"{results_df['candidates_25k'].max():,}"
)

print(
    f"50k maximum              : "
    f"{results_df['candidates_50k'].max():,}"
)


# ============================================================
# ADDITIONAL RECOVERY
# ============================================================

current_total = int(
    scored[
        "current_captured"
    ].sum()
)

captured_25k_total = int(
    scored[
        "captured_25k"
    ].sum()
)

captured_50k_total = int(
    scored[
        "captured_50k"
    ].sum()
)


print("\n----- ADDITIONAL TRUE-MATCH RECOVERY -----")

print(
    f"Current V2 captured       : "
    f"{current_total:,}"
)

print(
    f"25k strategy captured     : "
    f"{captured_25k_total:,}"
)

print(
    f"Additional with 25k       : "
    f"{captured_25k_total - current_total:,}"
)

print(
    f"50k strategy captured     : "
    f"{captured_50k_total:,}"
)

print(
    f"Additional with 50k       : "
    f"{captured_50k_total - current_total:,}"
)


# ============================================================
# FULL-SCALE ESTIMATE
# ============================================================

mean_25k = results_df[
    "candidates_25k"
].mean()

mean_50k = results_df[
    "candidates_50k"
].mean()


estimated_25k = (
    mean_25k
    * len(source1)
)

estimated_50k = (
    mean_50k
    * len(source1)
)


print("\n" + "=" * 80)
print("ROUGH FULL-SCALE ESTIMATE")
print("=" * 80)


print(
    f"Current V2 estimated pairs: "
    f"{baseline_mean * len(source1):,.0f}"
)

print(
    f"25k estimated pairs       : "
    f"{estimated_25k:,.0f}"
)

print(
    f"50k estimated pairs       : "
    f"{estimated_50k:,.0f}"
)


# ============================================================
# RECOVERY / COST
# ============================================================

additional_25k = (
    captured_25k_total
    - current_total
)

additional_50k = (
    captured_50k_total
    - current_total
)


print("\n----- TRADE-OFF -----")


if additional_25k > 0:

    print(
        f"25k: "
        f"{additional_25k:,} additional true pairs "
        f"for "
        f"{mean_25k / baseline_mean:.2f}x "
        f"the baseline candidate volume"
    )

else:

    print(
        "25k: no additional true pairs recovered."
    )


if additional_50k > 0:

    print(
        f"50k: "
        f"{additional_50k:,} additional true pairs "
        f"for "
        f"{mean_50k / baseline_mean:.2f}x "
        f"the baseline candidate volume"
    )

else:

    print(
        "50k: no additional true pairs recovered."
    )


# ============================================================
# WORST RECALL
# ============================================================

print("\n" + "=" * 80)
print("WORST RECALL — 25K STRATEGY")
print("=" * 80)

print(
    scored
    .sort_values(
        [
            "recall_25k",
            "candidates_25k"
        ]
    )
    .head(15)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("WORST RECALL — 50K STRATEGY")
print("=" * 80)

print(
    scored
    .sort_values(
        [
            "recall_50k",
            "candidates_50k"
        ]
    )
    .head(15)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    OUTPUT_DIR
    / "benchmark_relaxed_address_blocking.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)

print(
    f"Rows scanned: "
    f"{total_rows_scanned:,}"
)

print(
    f"Total runtime: "
    f"{time.time() - overall_start:.1f}s"
)

print(
    "Results saved to:"
)

print(
    output_path
)