from pathlib import Path
from collections import defaultdict
import json
import time

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

NORMALIZED_DIR = PROJECT_ROOT / "output" / "normalized"

SOURCE1_PATH = NORMALIZED_DIR / "train_source1_normalized.parquet"
SOURCE2_PATH = NORMALIZED_DIR / "train_source2_normalized.parquet"
SOURCE3_PATH = NORMALIZED_DIR / "train_source3_normalized.parquet"

GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.parquet"
)

BASELINE_RESULTS_PATH = (
    PROJECT_ROOT
    / "output"
    / "benchmark_relaxed_address_blocking.csv"
)

TOKEN_STATS_DIR = (
    PROJECT_ROOT
    / "code"
    / "business_entity_resolution"
    / "data"
    / "token_stats"
)

NAME_COUNTS_PATH = TOKEN_STATS_DIR / "name_token_counts.json"
ADDRESS_COUNTS_PATH = TOKEN_STATS_DIR / "address_token_counts.json"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "benchmark_selective_address_number_25k.csv"
)

# ============================================================
# SETTINGS
# ============================================================

CURRENT_MAX_FREQ = 10_000

# New selective rule:
# address token frequency > 10,000 and <= 25,000
SELECTIVE_MIN_FREQ = 10_000
SELECTIVE_MAX_FREQ = 25_000

BATCH_SIZE = 100_000


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_country(value):
    return safe_text(value).casefold()


def get_tokens(value):
    value = safe_text(value)

    if not value:
        return set()

    return set(value.split())


def rare_tokens(tokens, counts, max_frequency):
    result = set()

    for token in tokens:
        frequency = counts.get(token, 10**18)

        if frequency <= max_frequency:
            result.add(token)

    return result


def rarest_two_tokens(tokens, counts):
    if len(tokens) < 2:
        return ()

    ordered = sorted(
        tokens,
        key=lambda token: (
            counts.get(token, 10**18),
            token
        )
    )

    return tuple(ordered[:2])


def make_pair(tokens):
    if len(tokens) != 2:
        return None

    return tuple(sorted(tokens))


def extract_numbers(address):
    """
    Extract standalone numeric address tokens.

    Example:
        'shop 304 rajkot' -> {'304'}
    """

    return {
        token
        for token in get_tokens(address)
        if token.isdigit()
    }


def load_filtered_parquet(path, columns, ids_column, ids):
    """
    Load only requested IDs from a Parquet file.
    """

    dataset = ds.dataset(
        str(path),
        format="parquet"
    )

    table = dataset.to_table(
        columns=columns,
        filter=pc.is_in(
            pc.field(ids_column),
            value_set=pa.array(ids)
        )
    )

    return table.to_pandas()


def parse_ground_truth(value):
    value = safe_text(value)

    if not value:
        return set()

    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


# ============================================================
# LOAD THE SAME 1,000-ENTITY SAMPLE
# ============================================================

if not BASELINE_RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Baseline benchmark not found:\n"
        f"{BASELINE_RESULTS_PATH}\n\n"
        "Run benchmark_relaxed_address_blocking.py first."
    )

baseline_df = pd.read_csv(
    BASELINE_RESULTS_PATH
)

sample_ids = (
    baseline_df["s1_id"]
    .astype(str)
    .tolist()
)

if len(sample_ids) != 1000:
    raise ValueError(
        f"Expected 1,000 benchmark entities, "
        f"but found {len(sample_ids):,}."
    )

print(
    f"Loaded benchmark sample: "
    f"{len(sample_ids):,} Source-1 entities"
)


# ============================================================
# LOAD TOKEN FREQUENCIES
# ============================================================

with open(
    NAME_COUNTS_PATH,
    "r",
    encoding="utf-8"
) as f:
    name_counts = json.load(f)

with open(
    ADDRESS_COUNTS_PATH,
    "r",
    encoding="utf-8"
) as f:
    address_counts = json.load(f)

print(
    f"Loaded {len(name_counts):,} name token frequencies"
)

print(
    f"Loaded {len(address_counts):,} address token frequencies"
)


# ============================================================
# LOAD SOURCE-1 SAMPLE
# ============================================================

source1 = load_filtered_parquet(
    SOURCE1_PATH,
    [
        "entity_id",
        "country",
        "norm_name",
        "compact_name",
        "norm_address"
    ],
    "entity_id",
    sample_ids
)

source1 = (
    source1
    .set_index("entity_id")
    .loc[sample_ids]
    .reset_index()
)

if len(source1) != len(sample_ids):
    raise ValueError(
        "Some Source-1 benchmark IDs were not found."
    )

print(
    f"Loaded Source-1 sample rows: "
    f"{len(source1):,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

ground_truth = load_filtered_parquet(
    GROUND_TRUTH_PATH,
    [
        "source1_entity_id",
        "matched_entity_ids"
    ],
    "source1_entity_id",
    sample_ids
)

ground_truth = (
    ground_truth
    .set_index("source1_entity_id")
    .reindex(sample_ids)
    .reset_index()
)


gt_by_index = []

for _, row in ground_truth.iterrows():

    gt_by_index.append(
        parse_ground_truth(
            row["matched_entity_ids"]
        )
    )


entities_with_matches = sum(
    1
    for matches in gt_by_index
    if matches
)

print(
    f"Entities with at least one true match: "
    f"{entities_with_matches:,}"
)


# ============================================================
# PREPARE SOURCE-1 QUERY INFORMATION
# ============================================================

query_country = []
query_number_tokens = []

for _, row in source1.iterrows():

    query_country.append(
        clean_country(row["country"])
    )

    query_number_tokens.append(
        extract_numbers(
            row["norm_address"]
        )
    )


# ============================================================
# BUILD CURRENT V2 LOOKUPS
# ============================================================

exact_name_lookup = defaultdict(list)
compact_name_lookup = defaultdict(list)

name_single_lookup = defaultdict(list)
name_pair_lookup = defaultdict(list)

address_single_lookup = defaultdict(list)
address_pair_lookup = defaultdict(list)

# New relaxed address lookup
selective_address_lookup = defaultdict(list)


for s1_index, row in source1.iterrows():

    name = safe_text(
        row["norm_name"]
    )

    compact_name = safe_text(
        row["compact_name"]
    )

    address = safe_text(
        row["norm_address"]
    )

    name_tokens = get_tokens(name)
    address_tokens = get_tokens(address)


    # --------------------------------------------------------
    # 1. Exact normalized name
    # --------------------------------------------------------

    if name:

        exact_name_lookup[
            name
        ].append(
            s1_index
        )


    # --------------------------------------------------------
    # 2. Compact name
    # --------------------------------------------------------

    if compact_name:

        compact_name_lookup[
            compact_name
        ].append(
            s1_index
        )


    # --------------------------------------------------------
    # 3. Rare name tokens <= 10k
    # --------------------------------------------------------

    for token in rare_tokens(
        name_tokens,
        name_counts,
        CURRENT_MAX_FREQ
    ):

        name_single_lookup[
            token
        ].append(
            s1_index
        )


    # --------------------------------------------------------
    # 4. Name pair
    # --------------------------------------------------------

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
    # 5. Rare address tokens <= 10k
    # --------------------------------------------------------

    for token in rare_tokens(
        address_tokens,
        address_counts,
        CURRENT_MAX_FREQ
    ):

        address_single_lookup[
            token
        ].append(
            s1_index
        )


    # --------------------------------------------------------
    # 6. Address pair
    # --------------------------------------------------------

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
    # 7. NEW SELECTIVE ADDRESS TOKENS
    #
    # Frequency:
    #     10,001 - 50,000
    #
    # These will NOT be used alone.
    # A shared number will also be required.
    # --------------------------------------------------------

    for token in address_tokens:

        frequency = address_counts.get(
            token,
            10**18
        )

        if (
            SELECTIVE_MIN_FREQ
            < frequency
            <= SELECTIVE_MAX_FREQ
        ):

            selective_address_lookup[
                token
            ].append(
                s1_index
            )


print("\n" + "=" * 70)
print("LOOKUP SIZES")
print("=" * 70)

print(
    f"Exact name keys          : "
    f"{len(exact_name_lookup):,}"
)

print(
    f"Compact name keys        : "
    f"{len(compact_name_lookup):,}"
)

print(
    f"Rare name keys           : "
    f"{len(name_single_lookup):,}"
)

print(
    f"Name pair keys           : "
    f"{len(name_pair_lookup):,}"
)

print(
    f"Rare address keys        : "
    f"{len(address_single_lookup):,}"
)

print(
    f"Address pair keys        : "
    f"{len(address_pair_lookup):,}"
)

print(
    f"Selective address keys   : "
    f"{len(selective_address_lookup):,}"
)


# ============================================================
# RESULT ARRAYS
# ============================================================

current_candidate_counts = [
    0
] * len(sample_ids)

selective_rule_candidate_counts = [
    0
] * len(sample_ids)

final_candidate_counts = [
    0
] * len(sample_ids)


current_captured = [
    set()
    for _ in sample_ids
]

final_captured = [
    set()
    for _ in sample_ids
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

total_rows_scanned = 0
overall_start = time.time()


for source_path, source_label in source_files:

    print("\n" + "=" * 70)
    print(
        f"SCANNING {source_label}"
    )
    print("=" * 70)

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

            entity_id = safe_text(
                row.entity_id
            )

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
            # CURRENT V2
            # =================================================

            current_matches = set()


            # 1. Exact normalized name

            if name:

                current_matches.update(
                    exact_name_lookup.get(
                        name,
                        ()
                    )
                )


            # 2. Compact name

            if compact_name:

                current_matches.update(
                    compact_name_lookup.get(
                        compact_name,
                        ()
                    )
                )


            # 3. Rare name tokens

            name_tokens = get_tokens(
                name
            )

            for token in rare_tokens(
                name_tokens,
                name_counts,
                CURRENT_MAX_FREQ
            ):

                current_matches.update(
                    name_single_lookup.get(
                        token,
                        ()
                    )
                )


            # 4. Name pair

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


            # 5. Rare address tokens

            address_tokens = get_tokens(
                address
            )

            for token in rare_tokens(
                address_tokens,
                address_counts,
                CURRENT_MAX_FREQ
            ):

                current_matches.update(
                    address_single_lookup.get(
                        token,
                        ()
                    )
                )


            # 6. Address pair

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


            # 7. Country filter

            current_matches = {
                s1_index
                for s1_index in current_matches
                if country
                == query_country[s1_index]
            }


            # =================================================
            # NEW SELECTIVE RULE
            #
            # Condition 1:
            # address token frequency = 10,001-50,000
            #
            # AND
            #
            # Condition 2:
            # at least one shared numeric address token
            # =================================================

            selective_address_matches = set()

            source_numbers = {
                token
                for token in address_tokens
                if token.isdigit()
            }


            if source_numbers:

                moderate_address_matches = set()


                # Find S1 records sharing a moderate-frequency
                # address token.

                for token in address_tokens:

                    frequency = address_counts.get(
                        token,
                        10**18
                    )

                    if (
                        SELECTIVE_MIN_FREQ
                        < frequency
                        <= SELECTIVE_MAX_FREQ
                    ):

                        moderate_address_matches.update(
                            selective_address_lookup.get(
                                token,
                                ()
                            )
                        )


                # Country must still agree.

                moderate_address_matches = {
                    s1_index
                    for s1_index
                    in moderate_address_matches
                    if country
                    == query_country[s1_index]
                }


                # Require shared number.

                for s1_index in moderate_address_matches:

                    if (
                        source_numbers
                        & query_number_tokens[s1_index]
                    ):

                        selective_address_matches.add(
                            s1_index
                        )


            # =================================================
            # FINAL CANDIDATES
            #
            # V2 + selective rule
            # =================================================

            final_matches = (
                current_matches
                | selective_address_matches
            )


            selective_only = (
                selective_address_matches
                - current_matches
            )


            # =================================================
            # CANDIDATE COUNTS
            # =================================================

            for s1_index in current_matches:

                current_candidate_counts[
                    s1_index
                ] += 1


            for s1_index in selective_only:

                selective_rule_candidate_counts[
                    s1_index
                ] += 1


            for s1_index in final_matches:

                final_candidate_counts[
                    s1_index
                ] += 1


            # =================================================
            # TRUE MATCH CAPTURE
            # =================================================

            for s1_index in current_matches:

                if entity_id in gt_by_index[s1_index]:

                    current_captured[
                        s1_index
                    ].add(
                        entity_id
                    )


            for s1_index in final_matches:

                if entity_id in gt_by_index[s1_index]:

                    final_captured[
                        s1_index
                    ].add(
                        entity_id
                    )


            total_rows_scanned += 1


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
# BUILD RESULT TABLE
# ============================================================

results = []


for s1_index, s1_id in enumerate(
    sample_ids
):

    true_matches = gt_by_index[
        s1_index
    ]

    true_count = len(
        true_matches
    )

    current_count = len(
        current_captured[
            s1_index
        ]
    )

    final_count = len(
        final_captured[
            s1_index
        ]
    )

    newly_recovered = (
        final_captured[s1_index]
        - current_captured[s1_index]
    )


    if true_count > 0:

        current_recall = (
            current_count
            / true_count
        )

        final_recall = (
            final_count
            / true_count
        )

    else:

        current_recall = pd.NA
        final_recall = pd.NA


    baseline_row = baseline_df[
        baseline_df["s1_id"]
        == s1_id
    ]


    if baseline_row.empty:

        baseline_candidates = pd.NA

    else:

        baseline_candidates = (
            baseline_row[
                "baseline_candidates"
            ].iloc[0]
        )


    results.append(
        {
            "s1_id":
                s1_id,

            "true_matches":
                true_count,

            "baseline_candidates":
                baseline_candidates,

            "recomputed_current_candidates":
                current_candidate_counts[
                    s1_index
                ],

            "selective_rule_only_candidates":
                selective_rule_candidate_counts[
                    s1_index
                ],

            "final_candidates":
                final_candidate_counts[
                    s1_index
                ],

            "current_captured":
                current_count,

            "final_captured":
                final_count,

            "newly_recovered":
                len(newly_recovered),

            "current_recall":
                current_recall,

            "final_recall":
                final_recall
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# BASELINE CONSISTENCY CHECK
# ============================================================

baseline_numeric = pd.to_numeric(
    results_df[
        "baseline_candidates"
    ],
    errors="coerce"
)

recomputed_numeric = pd.to_numeric(
    results_df[
        "recomputed_current_candidates"
    ],
    errors="coerce"
)

differences = (
    recomputed_numeric
    - baseline_numeric
)

max_abs_difference = (
    differences.abs().max()
)


print("\n" + "=" * 70)
print("BASELINE CONSISTENCY CHECK")
print("=" * 70)

print(
    f"Maximum absolute candidate-count difference: "
    f"{max_abs_difference}"
)


if (
    pd.notna(max_abs_difference)
    and max_abs_difference != 0
):

    print(
        "\nWARNING:"
    )

    print(
        "Recomputed V2 candidate counts do not "
        "exactly match the saved benchmark."
    )

    print(
        "Do not treat the new candidate-volume "
        "comparison as final until this is checked."
    )


# ============================================================
# SUMMARY
# ============================================================

scored = results_df[
    results_df["true_matches"] > 0
].copy()


current_recall_mean = (
    scored[
        "current_recall"
    ].mean()
)

final_recall_mean = (
    scored[
        "final_recall"
    ].mean()
)


total_current_captured = int(
    scored[
        "current_captured"
    ].sum()
)

total_final_captured = int(
    scored[
        "final_captured"
    ].sum()
)


additional_recovered = (
    total_final_captured
    - total_current_captured
)


mean_current_candidates = (
    results_df[
        "recomputed_current_candidates"
    ].mean()
)

mean_selective_only_candidates = (
    results_df[
        "selective_rule_only_candidates"
    ].mean()
)

mean_final_candidates = (
    results_df[
        "final_candidates"
    ].mean()
)


print("\n" + "=" * 70)
print(
    "SELECTIVE ADDRESS + NUMBER BENCHMARK"
)
print("=" * 70)


print(
    f"Entities evaluated       : "
    f"{len(results_df):,}"
)

print(
    f"Entities with true match : "
    f"{len(scored):,}"
)


print("\n----- BLOCKING RECALL -----")

print(
    f"Current V2 recall          : "
    f"{current_recall_mean:.4%}"
)

print(
    f"V2 + selective rule recall : "
    f"{final_recall_mean:.4%}"
)


print("\n----- CANDIDATE SET SIZE -----")

print(
    f"Current V2 mean            : "
    f"{mean_current_candidates:,.2f}"
)

print(
    f"Selective-rule-only mean   : "
    f"{mean_selective_only_candidates:,.2f}"
)

print(
    f"Final V2 + selective mean  : "
    f"{mean_final_candidates:,.2f}"
)


print(
    f"\nCurrent V2 maximum         : "
    f"{results_df['recomputed_current_candidates'].max():,.0f}"
)

print(
    f"Selective-rule-only max    : "
    f"{results_df['selective_rule_only_candidates'].max():,.0f}"
)

print(
    f"Final maximum              : "
    f"{results_df['final_candidates'].max():,.0f}"
)


print("\n----- TRUE MATCH RECOVERY -----")

print(
    f"Current V2 captured        : "
    f"{total_current_captured:,}"
)

print(
    f"Final captured             : "
    f"{total_final_captured:,}"
)

print(
    f"Additional true matches    : "
    f"{additional_recovered:,}"
)


print("\n----- CANDIDATE COST -----")

if mean_current_candidates > 0:

    print(
        f"Final / current ratio      : "
        f"{mean_final_candidates / mean_current_candidates:.2f}x"
    )

    print(
        f"Additional candidates/S1   : "
        f"{mean_selective_only_candidates:,.2f}"
    )


# ============================================================
# ROUGH FULL-SCALE ESTIMATE
# ============================================================

FULL_SOURCE1_COUNT = 2_206_821

estimated_final_pairs = (
    mean_final_candidates
    * FULL_SOURCE1_COUNT
)

print(
    "\n----- ROUGH FULL-SCALE ESTIMATE -----"
)

print(
    f"Estimated final candidates : "
    f"{estimated_final_pairs:,.0f}"
)


# ============================================================
# ENTITIES WITH NEWLY RECOVERED TRUE MATCHES
# ============================================================

recovered_rows = results_df[
    results_df["newly_recovered"] > 0
]


print(
    "\n" + "=" * 70
)

print(
    "ENTITIES WITH NEWLY RECOVERED TRUE MATCHES"
)

print(
    "=" * 70
)


if recovered_rows.empty:

    print(
        "No additional true matches recovered."
    )

else:

    print(
        recovered_rows[
            [
                "s1_id",
                "true_matches",
                "current_captured",
                "final_captured",
                "newly_recovered",
                "recomputed_current_candidates",
                "selective_rule_only_candidates",
                "final_candidates",
                "current_recall",
                "final_recall"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\n" + "=" * 70
)

print(
    "BENCHMARK COMPLETE"
)

print(
    "=" * 70
)

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
    OUTPUT_PATH
)