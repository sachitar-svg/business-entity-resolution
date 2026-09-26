import json
import time
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

# Thresholds we want to compare
THRESHOLDS = [
    50,
    100,
    250,
    500,
    1000,
    2000,
    5000,
    10000
]


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
# HELPERS
# ============================================================

def safe_text(value):
    """
    Convert a value safely to a stripped string.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_country(value):
    """
    Normalize country values for comparison.
    """

    return safe_text(value).casefold()


def get_tokens(value):
    """
    Return unique tokens of length >= 2.

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
    threshold
):
    """
    Keep tokens whose global frequency is
    <= threshold.
    """

    return {
        token
        for token in tokens
        if counts.get(
            token,
            10**18
        ) <= threshold
    }


def rarest_two_tokens(
    tokens,
    counts
):
    """
    Return the two least-frequent tokens.

    This is the same concept used by the
    current V2 name/address pair rules.
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


def compact_name(value):
    """
    The normalized Parquet already contains compact_name,
    so this helper is mainly for consistency.
    """

    return safe_text(value)


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

sample_ids = set(
    s1_sample["entity_id"]
)

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

gt_lookup = {}

for row in ground_truth.itertuples(
    index=False
):

    s1_id = row.source1_entity_id

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


# ============================================================
# GET ALL TRUE MATCH IDS FOR SAMPLE
# ============================================================

true_source2_ids = set()
true_source3_ids = set()

for matches in gt_lookup.values():

    for entity_id in matches:

        if entity_id.startswith("S2-"):
            true_source2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            true_source3_ids.add(entity_id)


print(
    f"Sample S1 entities       : {len(sample_ids):,}"
)

print(
    f"True Source 2 IDs needed : {len(true_source2_ids):,}"
)

print(
    f"True Source 3 IDs needed : {len(true_source3_ids):,}"
)

print(
    f"Total true pairs         : "
    f"{len(true_source2_ids) + len(true_source3_ids):,}"
)


# ============================================================
# LOAD ONLY THE TRUE MATCH RECORDS
# ============================================================

def load_target_records(
    parquet_path,
    wanted_ids,
    source_label
):
    """
    Scan a normalized Parquet file in batches but retain
    only the records whose entity_id occurs in wanted_ids.

    This means we do NOT build the candidate set.
    """

    print("\n" + "-" * 70)
    print(f"Searching {source_label}")
    print("-" * 70)

    if not wanted_ids:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "country",
                "norm_name",
                "compact_name",
                "norm_address"
            ]
        )

    parquet_file = pq.ParquetFile(
        parquet_path
    )

    found_parts = []

    found_ids = set()

    rows_scanned = 0

    start_time = time.time()

    columns = [
        "entity_id",
        "country",
        "norm_name",
        "compact_name",
        "norm_address"
    ]

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=columns
        ),
        start=1
    ):

        chunk = batch.to_pandas()

        rows_scanned += len(chunk)

        mask = chunk["entity_id"].isin(
            wanted_ids
        )

        selected = chunk.loc[
            mask
        ]

        if not selected.empty:

            found_parts.append(
                selected.copy()
            )

            found_ids.update(
                selected["entity_id"]
            )

        # Once every required ID has been found,
        # there is no reason to scan further.
        if found_ids >= wanted_ids:
            print(
                f"  All {len(wanted_ids):,} target IDs found."
            )
            break

        if batch_number % 10 == 0:

            print(
                f"  Batch {batch_number:>3}"
                f" | scanned {rows_scanned:,}"
                f" | found {len(found_ids):,}/"
                f"{len(wanted_ids):,}"
            )

        del chunk

    if found_parts:

        result = pd.concat(
            found_parts,
            ignore_index=True
        )

    else:

        result = pd.DataFrame(
            columns=columns
        )

    elapsed = time.time() - start_time

    missing = wanted_ids - found_ids

    print(
        f"Finished {source_label} in {elapsed:.1f}s"
    )

    print(
        f"Rows scanned : {rows_scanned:,}"
    )

    print(
        f"IDs found    : {len(found_ids):,}"
    )

    print(
        f"IDs missing  : {len(missing):,}"
    )

    if missing:

        print(
            "WARNING: Some ground-truth IDs were not found."
        )

        print(
            list(sorted(missing))[:20]
        )

    return result


source2_true_records = load_target_records(
    SOURCE2_PATH,
    true_source2_ids,
    "Source 2"
)

source3_true_records = load_target_records(
    SOURCE3_PATH,
    true_source3_ids,
    "Source 3"
)


# ============================================================
# COMBINE TRUE MATCH RECORDS
# ============================================================

true_source_records = pd.concat(
    [
        source2_true_records,
        source3_true_records
    ],
    ignore_index=True
)


true_record_lookup = {}

for row in true_source_records.itertuples(
    index=False
):

    true_record_lookup[
        row.entity_id
    ] = row


print("\n" + "=" * 80)
print("TRUE MATCH RECORDS LOADED")
print("=" * 80)

print(
    f"Records retrieved: "
    f"{len(true_record_lookup):,}"
)


# ============================================================
# EVALUATE ONE TRUE PAIR
# ============================================================

def evaluate_pair(
    s1_row,
    target_row,
    name_threshold,
    address_threshold
):
    """
    Determine whether a TRUE Source 1 -> Source 2/3
    pair would be retained by the V2 blocking rules
    at the specified single-token threshold.

    This function evaluates ONLY the real ground-truth
    pairs, not all possible candidate pairs.
    """

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    s1_country = clean_country(
        s1_row.country
    )

    target_country = clean_country(
        target_row.country
    )

    if s1_country != target_country:
        return False, "country_mismatch"


    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    s1_name = safe_text(
        s1_row.norm_name
    )

    target_name = safe_text(
        target_row.norm_name
    )

    s1_compact = safe_text(
        s1_row.compact_name
    )

    target_compact = safe_text(
        target_row.compact_name
    )

    s1_name_tokens = get_tokens(
        s1_name
    )

    target_name_tokens = get_tokens(
        target_name
    )


    # --------------------------------------------------------
    # 1. Exact normalized name
    # --------------------------------------------------------

    if (
        s1_name
        and target_name
        and s1_name == target_name
    ):

        return True, "exact_name"


    # --------------------------------------------------------
    # 2. Compact name
    # --------------------------------------------------------

    if (
        s1_compact
        and target_compact
        and s1_compact == target_compact
    ):

        return True, "compact_name"


    # --------------------------------------------------------
    # 3. Rare name token
    # --------------------------------------------------------

    s1_rare_name = rare_tokens(
        s1_name_tokens,
        name_counts,
        name_threshold
    )

    target_rare_name = rare_tokens(
        target_name_tokens,
        name_counts,
        name_threshold
    )

    if (
        s1_rare_name
        & target_rare_name
    ):

        return True, "rare_name_token"


    # --------------------------------------------------------
    # 4. Name pair
    # --------------------------------------------------------

    s1_name_pair = make_pair(
        rarest_two_tokens(
            s1_name_tokens,
            name_counts
        )
    )

    target_name_pair = make_pair(
        rarest_two_tokens(
            target_name_tokens,
            name_counts
        )
    )

    if (
        s1_name_pair
        and target_name_pair
        and s1_name_pair == target_name_pair
    ):

        return True, "name_pair"


    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    s1_address = safe_text(
        s1_row.norm_address
    )

    target_address = safe_text(
        target_row.norm_address
    )

    s1_address_tokens = get_tokens(
        s1_address
    )

    target_address_tokens = get_tokens(
        target_address
    )


    # --------------------------------------------------------
    # 5. Rare address token
    # --------------------------------------------------------

    s1_rare_address = rare_tokens(
        s1_address_tokens,
        address_counts,
        address_threshold
    )

    target_rare_address = rare_tokens(
        target_address_tokens,
        address_counts,
        address_threshold
    )

    if (
        s1_rare_address
        & target_rare_address
    ):

        return True, "rare_address_token"


    # --------------------------------------------------------
    # 6. Address pair
    # --------------------------------------------------------

    s1_address_pair = make_pair(
        rarest_two_tokens(
            s1_address_tokens,
            address_counts
        )
    )

    target_address_pair = make_pair(
        rarest_two_tokens(
            target_address_tokens,
            address_counts
        )
    )

    if (
        s1_address_pair
        and target_address_pair
        and s1_address_pair == target_address_pair
    ):

        return True, "address_pair"


    # --------------------------------------------------------
    # Nothing matched
    # --------------------------------------------------------

    return False, "no_blocking_key"


# ============================================================
# PREPARE TRUE PAIR LIST
# ============================================================

sample_rows = {
    row.entity_id: row
    for row in s1_sample.itertuples(
        index=False
    )
}


true_pairs = []

missing_target_records = 0

for s1_id, true_ids in gt_lookup.items():

    s1_row = sample_rows[s1_id]

    for target_id in true_ids:

        target_row = true_record_lookup.get(
            target_id
        )

        if target_row is None:

            missing_target_records += 1

            continue

        true_pairs.append(
            (
                s1_row,
                target_row
            )
        )


print("\n" + "=" * 80)
print("TRUE PAIR EVALUATION SET")
print("=" * 80)

print(
    f"True pairs available: "
    f"{len(true_pairs):,}"
)

print(
    f"Missing target records skipped: "
    f"{missing_target_records:,}"
)


# ============================================================
# THRESHOLD EXPERIMENT
# ============================================================

results = []

overall_start = time.time()


for threshold in THRESHOLDS:

    threshold_start = time.time()

    captured = 0

    reason_counts = {}

    missed_examples = []

    for s1_row, target_row in true_pairs:

        captured_flag, reason = evaluate_pair(
            s1_row,
            target_row,
            threshold,
            threshold
        )

        if captured_flag:

            captured += 1

            reason_counts[reason] = (
                reason_counts.get(
                    reason,
                    0
                ) + 1
            )

        else:

            if len(missed_examples) < 10:

                missed_examples.append(
                    (
                        s1_row.entity_id,
                        target_row.entity_id
                    )
                )


    total_pairs = len(true_pairs)

    if total_pairs:

        recall = (
            captured
            / total_pairs
        )

    else:

        recall = 0.0


    results.append(
        {
            "threshold": threshold,
            "true_pairs": total_pairs,
            "captured": captured,
            "missed": total_pairs - captured,
            "recall": recall,
            "exact_name": reason_counts.get(
                "exact_name",
                0
            ),
            "compact_name": reason_counts.get(
                "compact_name",
                0
            ),
            "rare_name_token": reason_counts.get(
                "rare_name_token",
                0
            ),
            "name_pair": reason_counts.get(
                "name_pair",
                0
            ),
            "rare_address_token": reason_counts.get(
                "rare_address_token",
                0
            ),
            "address_pair": reason_counts.get(
                "address_pair",
                0
            ),
            "runtime_seconds": (
                time.time()
                - threshold_start
            )
        }
    )


    print(
        f"\nThreshold {threshold:>5}"
        f" | recall: {recall:.4%}"
        f" | captured: {captured:,}/{total_pairs:,}"
        f" | missed: {total_pairs - captured:,}"
    )

    if missed_examples:

        print(
            "  Example missed pairs:"
        )

        for s1_id, target_id in missed_examples[:5]:

            print(
                f"    {s1_id} -> {target_id}"
            )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("RARE-TOKEN THRESHOLD EXPERIMENT")
print("=" * 80)

print(
    results_df[
        [
            "threshold",
            "true_pairs",
            "captured",
            "missed",
            "recall",
            "exact_name",
            "compact_name",
            "rare_name_token",
            "name_pair",
            "rare_address_token",
            "address_pair"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# BEST RECALL BY THRESHOLD
# ============================================================

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

best_recall = results_df[
    "recall"
].max()

best_rows = results_df[
    results_df["recall"] == best_recall
]

print(
    f"Highest recall observed: "
    f"{best_recall:.4%}"
)

print(
    "Threshold(s) achieving highest recall:"
)

print(
    best_rows[
        "threshold"
    ].tolist()
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    OUTPUT_DIR
    / "benchmark_rare_token_thresholds.csv"
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
    f"Total runtime: "
    f"{time.time() - overall_start:.1f}s"
)

print(
    f"Results saved to:"
)

print(
    output_path
)