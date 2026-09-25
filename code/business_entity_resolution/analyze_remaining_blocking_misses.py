import re
import json
from pathlib import Path

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

BLOCKING_RESULTS_PATH = (
    OUTPUT_DIR
    / "benchmark_address_two_rare_tokens.csv"
)

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

NAME_STATS_PATH = (
    TOKEN_STATS_DIR
    / "name_token_counts.json"
)

ADDRESS_STATS_PATH = (
    TOKEN_STATS_DIR
    / "address_token_counts.json"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 100_000

CURRENT_THRESHOLD = 10_000


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
    Return unique tokens with length >= 2.
    """

    value = safe_text(value)

    if not value:
        return set()

    return {
        token
        for token in value.split()
        if len(token) >= 2
    }


def get_numbers(value):
    """
    Extract numeric components from normalized address.
    """

    value = safe_text(value)

    return set(
        re.findall(
            r"\d+",
            value
        )
    )


def token_frequencies(
    tokens,
    counts
):
    """
    Return token -> global frequency mapping.
    """

    return {
        token: counts.get(
            token,
            10**18
        )
        for token in tokens
    }


def format_token_frequencies(
    mapping
):
    """
    Format token frequencies for CSV output.
    """

    if not mapping:
        return ""

    items = sorted(
        mapping.items(),
        key=lambda x: (
            x[1],
            x[0]
        )
    )

    return "; ".join(
        f"{token}:{freq:,}"
        for token, freq in items
    )


def load_records_by_ids(
    parquet_path,
    wanted_ids,
    label
):
    """
    Read a normalized Parquet file in batches and return
    only the requested entity IDs.
    """

    print("\n" + "-" * 70)
    print(f"Searching {label}")
    print("-" * 70)

    if not wanted_ids:
        return {}

    parquet_file = pq.ParquetFile(
        parquet_path
    )

    columns = [
        "entity_id",
        "country",
        "norm_name",
        "compact_name",
        "norm_address"
    ]

    found = {}

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=columns
        ),
        start=1
    ):

        chunk = batch.to_pandas()

        selected = chunk[
            chunk["entity_id"].isin(
                wanted_ids
            )
        ]

        for row in selected.itertuples(
            index=False
        ):

            found[row.entity_id] = row

        if (
            len(found) == len(wanted_ids)
        ):

            print(
                f"All {len(wanted_ids):,} IDs found."
            )

            break

        if batch_number % 10 == 0:

            print(
                f"  Batch {batch_number:>3}"
                f" | found {len(found):,}/"
                f"{len(wanted_ids):,}"
            )

        del chunk

    missing = (
        wanted_ids
        - set(found.keys())
    )

    print(
        f"{label} IDs found : "
        f"{len(found):,}"
    )

    print(
        f"{label} IDs missing: "
        f"{len(missing):,}"
    )

    return found


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
# LOAD PREVIOUS BLOCKING RESULTS
# ============================================================

print("\n" + "=" * 80)
print("LOADING PREVIOUS BLOCKING RESULTS")
print("=" * 80)

blocking_df = pd.read_csv(
    BLOCKING_RESULTS_PATH
)

print(
    f"Total true pairs in file: "
    f"{len(blocking_df):,}"
)


# Current V2 missed pairs

missed_df = blocking_df[
    blocking_df["current_v2"] == False
].copy()


print(
    f"Current V2 missed pairs : "
    f"{len(missed_df):,}"
)


# ============================================================
# LOAD SOURCE 1 RECORDS INVOLVED IN MISSES
# ============================================================

missed_s1_ids = set(
    missed_df["s1_id"]
)

missed_target_ids = set(
    missed_df["target_id"]
)

source1_records = load_records_by_ids(
    SOURCE1_PATH,
    missed_s1_ids,
    "Source 1"
)


# ============================================================
# SEPARATE S2 / S3 TARGET IDS
# ============================================================

missed_s2_ids = {
    entity_id
    for entity_id in missed_target_ids
    if entity_id.startswith("S2-")
}

missed_s3_ids = {
    entity_id
    for entity_id in missed_target_ids
    if entity_id.startswith("S3-")
}


# ============================================================
# LOAD TARGET RECORDS
# ============================================================

source2_records = load_records_by_ids(
    SOURCE2_PATH,
    missed_s2_ids,
    "Source 2"
)

source3_records = load_records_by_ids(
    SOURCE3_PATH,
    missed_s3_ids,
    "Source 3"
)


target_records = {}

target_records.update(
    source2_records
)

target_records.update(
    source3_records
)


# ============================================================
# ANALYZE EACH MISSED PAIR
# ============================================================

analysis_rows = []


for row in missed_df.itertuples(
    index=False
):

    s1_id = row.s1_id
    target_id = row.target_id

    s1 = source1_records.get(
        s1_id
    )

    target = target_records.get(
        target_id
    )

    if s1 is None or target is None:
        continue


    # --------------------------------------------------------
    # Basic values
    # --------------------------------------------------------

    s1_country = clean_country(
        s1.country
    )

    target_country = clean_country(
        target.country
    )

    same_country = (
        s1_country == target_country
    )


    # --------------------------------------------------------
    # Name tokens
    # --------------------------------------------------------

    s1_name_tokens = get_tokens(
        s1.norm_name
    )

    target_name_tokens = get_tokens(
        target.norm_name
    )

    shared_name_tokens = (
        s1_name_tokens
        & target_name_tokens
    )


    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    s1_address_tokens = get_tokens(
        s1.norm_address
    )

    target_address_tokens = get_tokens(
        target.norm_address
    )

    shared_address_tokens = (
        s1_address_tokens
        & target_address_tokens
    )


    # --------------------------------------------------------
    # Shared numeric components
    # --------------------------------------------------------

    s1_numbers = get_numbers(
        s1.norm_address
    )

    target_numbers = get_numbers(
        target.norm_address
    )

    shared_numbers = (
        s1_numbers
        & target_numbers
    )


    # --------------------------------------------------------
    # Token frequencies
    # --------------------------------------------------------

    shared_name_frequency = token_frequencies(
        shared_name_tokens,
        name_counts
    )

    shared_address_frequency = token_frequencies(
        shared_address_tokens,
        address_counts
    )


    # --------------------------------------------------------
    # Rare shared tokens under current threshold
    # --------------------------------------------------------

    rare_shared_name = {
        token
        for token, freq
        in shared_name_frequency.items()
        if freq <= CURRENT_THRESHOLD
    }

    rare_shared_address = {
        token
        for token, freq
        in shared_address_frequency.items()
        if freq <= CURRENT_THRESHOLD
    }


    # --------------------------------------------------------
    # Minimum frequency among shared tokens
    # --------------------------------------------------------

    if shared_name_frequency:

        min_shared_name_frequency = min(
            shared_name_frequency.values()
        )

    else:

        min_shared_name_frequency = None


    if shared_address_frequency:

        min_shared_address_frequency = min(
            shared_address_frequency.values()
        )

    else:

        min_shared_address_frequency = None


    # --------------------------------------------------------
    # Maximum frequency among shared tokens
    # --------------------------------------------------------

    if shared_address_frequency:

        max_shared_address_frequency = max(
            shared_address_frequency.values()
        )

    else:

        max_shared_address_frequency = None


    # --------------------------------------------------------
    # How many shared tokens are <= thresholds?
    # --------------------------------------------------------

    address_count_10k = sum(
        freq <= 10_000
        for freq
        in shared_address_frequency.values()
    )

    address_count_25k = sum(
        freq <= 25_000
        for freq
        in shared_address_frequency.values()
    )

    address_count_50k = sum(
        freq <= 50_000
        for freq
        in shared_address_frequency.values()
    )

    address_count_100k = sum(
        freq <= 100_000
        for freq
        in shared_address_frequency.values()
    )

    address_count_250k = sum(
        freq <= 250_000
        for freq
        in shared_address_frequency.values()
    )


    # --------------------------------------------------------
    # Main interpretation category
    # --------------------------------------------------------

    if not same_country:

        category = "country_mismatch"

    elif rare_shared_name:

        category = "name_token_above_or_within_rule"

    elif rare_shared_address:

        category = "address_token_above_or_within_rule"

    elif shared_address_tokens:

        category = "shared_address_tokens_but_all_too_common"

    elif shared_name_tokens:

        category = "shared_name_tokens_but_all_too_common"

    else:

        category = "no_shared_name_or_address_tokens"


    # --------------------------------------------------------
    # Save row
    # --------------------------------------------------------

    analysis_rows.append({

        "s1_id": s1_id,
        "target_id": target_id,

        "country": s1_country,
        "same_country": same_country,

        "s1_name": safe_text(
            s1.norm_name
        ),

        "target_name": safe_text(
            target.norm_name
        ),

        "s1_address": safe_text(
            s1.norm_address
        ),

        "target_address": safe_text(
            target.norm_address
        ),

        "shared_name_count": len(
            shared_name_tokens
        ),

        "shared_name_tokens": ",".join(
            sorted(shared_name_tokens)
        ),

        "shared_name_frequencies": (
            format_token_frequencies(
                shared_name_frequency
            )
        ),

        "rare_shared_name_count": len(
            rare_shared_name
        ),

        "shared_address_count": len(
            shared_address_tokens
        ),

        "shared_address_tokens": ",".join(
            sorted(shared_address_tokens)
        ),

        "shared_address_frequencies": (
            format_token_frequencies(
                shared_address_frequency
            )
        ),

        "rare_shared_address_count": len(
            rare_shared_address
        ),

        "shared_numbers_count": len(
            shared_numbers
        ),

        "shared_numbers": ",".join(
            sorted(shared_numbers)
        ),

        "min_shared_name_frequency":
            min_shared_name_frequency,

        "min_shared_address_frequency":
            min_shared_address_frequency,

        "max_shared_address_frequency":
            max_shared_address_frequency,

        "address_tokens_at_10k":
            address_count_10k,

        "address_tokens_at_25k":
            address_count_25k,

        "address_tokens_at_50k":
            address_count_50k,

        "address_tokens_at_100k":
            address_count_100k,

        "address_tokens_at_250k":
            address_count_250k,

        "category": category
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

analysis_df = pd.DataFrame(
    analysis_rows
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("REMAINING BLOCKING MISS ANALYSIS")
print("=" * 80)

print(
    f"\nMissed true pairs analyzed: "
    f"{len(analysis_df):,}"
)


# ------------------------------------------------------------
# Shared token summary
# ------------------------------------------------------------

print("\n----- SHARED NAME TOKENS -----")

print(
    f"Pairs with shared name token(s): "
    f"{(
        analysis_df['shared_name_count'] > 0
    ).sum():,}"
)


print("\n----- SHARED ADDRESS TOKENS -----")

print(
    f"Pairs with shared address token(s): "
    f"{(
        analysis_df['shared_address_count'] > 0
    ).sum():,}"
)


print("\n----- SHARED NUMBERS -----")

print(
    f"Pairs with shared address numbers: "
    f"{(
        analysis_df['shared_numbers_count'] > 0
    ).sum():,}"
)


# ------------------------------------------------------------
# Rare address token summary
# ------------------------------------------------------------

print("\n----- RARE SHARED ADDRESS TOKENS -----")

print(
    f"Pairs with shared address token "
    f"<= 10,000 frequency: "
    f"{(
        analysis_df['rare_shared_address_count'] > 0
    ).sum():,}"
)


print(
    f"Pairs with NO shared address token "
    f"<= 10,000: "
    f"{(
        analysis_df['rare_shared_address_count'] == 0
    ).sum():,}"
)


# ------------------------------------------------------------
# Threshold recovery potential
# ------------------------------------------------------------

print("\n----- POTENTIAL RECOVERY BY ADDRESS TOKEN THRESHOLD -----")

for threshold, column in [
    (10_000, "address_tokens_at_10k"),
    (25_000, "address_tokens_at_25k"),
    (50_000, "address_tokens_at_50k"),
    (100_000, "address_tokens_at_100k"),
    (250_000, "address_tokens_at_250k"),
]:

    recovered = int(
        (
            analysis_df[column] > 0
        ).sum()
    )

    print(
        f"Threshold <= {threshold:>6,}: "
        f"{recovered:>3} / "
        f"{len(analysis_df):,} missed pairs "
        f"would have at least one shared address token"
    )


# ------------------------------------------------------------
# Frequency statistics
# ------------------------------------------------------------

address_frequency_series = (
    analysis_df[
        "min_shared_address_frequency"
    ]
    .dropna()
)

if not address_frequency_series.empty:

    print(
        "\n----- MINIMUM SHARED ADDRESS TOKEN FREQUENCY -----"
    )

    print(
        f"Minimum : "
        f"{address_frequency_series.min():,.0f}"
    )

    print(
        f"Median  : "
        f"{address_frequency_series.median():,.0f}"
    )

    print(
        f"75th pct: "
        f"{address_frequency_series.quantile(0.75):,.0f}"
    )

    print(
        f"90th pct: "
        f"{address_frequency_series.quantile(0.90):,.0f}"
    )

    print(
        f"95th pct: "
        f"{address_frequency_series.quantile(0.95):,.0f}"
    )

    print(
        f"Maximum : "
        f"{address_frequency_series.max():,.0f}"
    )


# ============================================================
# CATEGORIES
# ============================================================

print("\n" + "=" * 80)
print("MISS CATEGORIES")
print("=" * 80)

print(
    analysis_df[
        "category"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# DETAILED MISSED PAIRS
# ============================================================

print("\n" + "=" * 80)
print("DETAILED MISSED PAIRS")
print("=" * 80)

display_columns = [
    "s1_id",
    "target_id",
    "shared_name_count",
    "shared_name_tokens",
    "shared_address_count",
    "shared_address_tokens",
    "shared_address_frequencies",
    "shared_numbers",
    "min_shared_address_frequency",
    "category"
]


print(
    analysis_df[
        display_columns
    ]
    .sort_values(
        [
            "shared_address_count",
            "min_shared_address_frequency"
        ],
        ascending=[
            False,
            True
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

output_path = (
    OUTPUT_DIR
    / "remaining_blocking_miss_analysis.csv"
)

analysis_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    "Detailed results saved to:"
)

print(
    output_path
)