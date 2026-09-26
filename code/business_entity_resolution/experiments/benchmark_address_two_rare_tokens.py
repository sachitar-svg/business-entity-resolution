import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PATHS
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
# SETTINGS
# ============================================================

SAMPLE_SIZE = 1000
RANDOM_STATE = 42

BATCH_SIZE = 100_000

SINGLE_TOKEN_MAX_FREQ = 10_000

MIN_SHARED_RARE_ADDRESS_TOKENS = 2


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
    Normalize country text for comparison.
    """

    return safe_text(value).casefold()


def get_tokens(value):
    """
    Return unique tokens of length >= 2.
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
    Keep only tokens whose global frequency is
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


def make_pair(tokens):
    """
    Create order-independent pair key.
    """

    if len(tokens) < 2:
        return None

    return "|".join(
        sorted(tokens)
    )


def rarest_two_tokens(tokens, counts):
    """
    Get the two least-frequent tokens.
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
# GET ALL TRUE TARGET IDS
# ============================================================

wanted_s2_ids = set()
wanted_s3_ids = set()

for matches in gt_lookup.values():

    for entity_id in matches:

        if entity_id.startswith("S2-"):

            wanted_s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):

            wanted_s3_ids.add(entity_id)


print(
    f"\nTrue Source 2 IDs: {len(wanted_s2_ids):,}"
)

print(
    f"True Source 3 IDs: {len(wanted_s3_ids):,}"
)


# ============================================================
# LOAD ONLY TRUE TARGET RECORDS
# ============================================================

def load_target_records(
    path,
    wanted_ids,
    label
):
    """
    Scan normalized Parquet in batches and retain
    only ground-truth target records.
    """

    print("\n" + "-" * 70)
    print(f"Searching {label}")
    print("-" * 70)

    if not wanted_ids:

        return {}

    parquet_file = pq.ParquetFile(
        path
    )

    found = {}

    columns = [
        "entity_id",
        "country",
        "norm_name",
        "compact_name",
        "norm_address"
    ]

    rows_scanned = 0

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=columns
        ),
        start=1
    ):

        chunk = batch.to_pandas()

        rows_scanned += len(chunk)

        selected = chunk[
            chunk["entity_id"].isin(wanted_ids)
        ]

        for row in selected.itertuples(
            index=False
        ):

            found[row.entity_id] = row

        if found.keys() >= wanted_ids:

            print(
                f"All {len(wanted_ids):,} target IDs found."
            )

            break

        if batch_number % 10 == 0:

            print(
                f"  Batch {batch_number:>3}"
                f" | scanned {rows_scanned:,}"
                f" | found {len(found):,}/"
                f"{len(wanted_ids):,}"
            )

        del chunk

    print(
        f"{label} records found: "
        f"{len(found):,}"
    )

    missing = wanted_ids - set(found.keys())

    if missing:

        print(
            f"WARNING: {len(missing)} IDs missing."
        )

        print(
            list(sorted(missing))[:20]
        )

    return found


source2_records = load_target_records(
    SOURCE2_PATH,
    wanted_s2_ids,
    "Source 2"
)

source3_records = load_target_records(
    SOURCE3_PATH,
    wanted_s3_ids,
    "Source 3"
)


target_lookup = {}

target_lookup.update(
    source2_records
)

target_lookup.update(
    source3_records
)


# ============================================================
# SOURCE 1 LOOKUP
# ============================================================

s1_lookup = {
    row.entity_id: row
    for row in s1_sample.itertuples(
        index=False
    )
}


# ============================================================
# EVALUATE CURRENT V2
# ============================================================

def current_v2_captures(
    s1_row,
    target_row
):
    """
    Check whether the existing V2 blocker would capture
    this true pair.
    """

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    if (
        clean_country(s1_row.country)
        != clean_country(target_row.country)
    ):
        return False, "country_mismatch"


    # --------------------------------------------------------
    # Names
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


    # 1. Exact normalized name

    if (
        s1_name
        and target_name
        and s1_name == target_name
    ):

        return True, "exact_name"


    # 2. Compact name

    if (
        s1_compact
        and target_compact
        and s1_compact == target_compact
    ):

        return True, "compact_name"


    # 3. Rare name token

    s1_rare_name = rare_tokens(
        s1_name_tokens,
        name_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    target_rare_name = rare_tokens(
        target_name_tokens,
        name_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    if (
        s1_rare_name
        & target_rare_name
    ):

        return True, "rare_name_token"


    # 4. Name pair

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
    # Addresses
    # --------------------------------------------------------

    s1_address_tokens = get_tokens(
        s1_row.norm_address
    )

    target_address_tokens = get_tokens(
        target_row.norm_address
    )


    # 5. Rare address token

    s1_rare_address = rare_tokens(
        s1_address_tokens,
        address_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    target_rare_address = rare_tokens(
        target_address_tokens,
        address_counts,
        SINGLE_TOKEN_MAX_FREQ
    )

    if (
        s1_rare_address
        & target_rare_address
    ):

        return True, "rare_address_token"


    # 6. Address pair

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


    return False, "no_blocking_key"


# ============================================================
# EVALUATE NEW ADDRESS RULE
# ============================================================

results = []

reason_counter = Counter()

total_pairs = 0


for s1_id, true_match_ids in gt_lookup.items():

    s1_row = s1_lookup[s1_id]

    for target_id in true_match_ids:

        target_row = target_lookup.get(
            target_id
        )

        if target_row is None:
            continue

        total_pairs += 1


        # ----------------------------------------------------
        # Current V2
        # ----------------------------------------------------

        current_captured, current_reason = (
            current_v2_captures(
                s1_row,
                target_row
            )
        )


        # ----------------------------------------------------
        # Rare address overlap
        # ----------------------------------------------------

        s1_address_tokens = get_tokens(
            s1_row.norm_address
        )

        target_address_tokens = get_tokens(
            target_row.norm_address
        )


        s1_rare_address = rare_tokens(
            s1_address_tokens,
            address_counts,
            SINGLE_TOKEN_MAX_FREQ
        )

        target_rare_address = rare_tokens(
            target_address_tokens,
            address_counts,
            SINGLE_TOKEN_MAX_FREQ
        )


        shared_rare_address = (
            s1_rare_address
            & target_rare_address
        )


        new_address_rule = (
            len(shared_rare_address)
            >= MIN_SHARED_RARE_ADDRESS_TOKENS
        )


        # Same country is required

        same_country = (
            clean_country(s1_row.country)
            == clean_country(target_row.country)
        )

        new_address_rule = (
            same_country
            and new_address_rule
        )


        # ----------------------------------------------------
        # Combined V2 + new address rule
        # ----------------------------------------------------

        combined_captured = (
            current_captured
            or new_address_rule
        )


        # New rule can recover only pairs missed by V2

        recovered = (
            not current_captured
            and new_address_rule
        )


        if current_captured:

            reason_counter[
                f"current_{current_reason}"
            ] += 1

        elif new_address_rule:

            reason_counter[
                "new_two_rare_address_tokens"
            ] += 1


        results.append({
            "s1_id": s1_id,
            "target_id": target_id,
            "current_v2": current_captured,
            "new_address_rule": new_address_rule,
            "combined": combined_captured,
            "shared_rare_address_tokens": len(
                shared_rare_address
            ),
            "recovered_by_new_rule": recovered
        })


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
print("TWO RARE ADDRESS TOKEN EXPERIMENT")
print("=" * 80)


print(
    f"\nTrue pairs evaluated:"
    f" {len(results_df):,}"
)


current_captured = int(
    results_df["current_v2"].sum()
)

new_rule_captured = int(
    results_df["new_address_rule"].sum()
)

combined_captured = int(
    results_df["combined"].sum()
)

recovered = int(
    results_df["recovered_by_new_rule"].sum()
)


print("\n----- PAIR CAPTURE -----")

print(
    f"Current V2 captured       : "
    f"{current_captured:,}"
)

print(
    f"New address rule captured : "
    f"{new_rule_captured:,}"
)

print(
    f"Combined captured         : "
    f"{combined_captured:,}"
)

print(
    f"Additional pairs recovered:"
    f" {recovered:,}"
)


# ============================================================
# RECALL
# ============================================================

if len(results_df) > 0:

    current_recall = (
        current_captured
        / len(results_df)
    )

    combined_recall = (
        combined_captured
        / len(results_df)
    )

else:

    current_recall = 0.0
    combined_recall = 0.0


print("\n----- RECALL -----")

print(
    f"Current V2 recall     : "
    f"{current_recall:.4%}"
)

print(
    f"Combined recall       : "
    f"{combined_recall:.4%}"
)

print(
    f"Recall improvement    : "
    f"{combined_recall - current_recall:.4%}"
)


# ============================================================
# DISTRIBUTION OF SHARED RARE ADDRESS TOKENS
# ============================================================

print("\n----- SHARED RARE ADDRESS TOKEN DISTRIBUTION -----")

print(
    results_df[
        "shared_rare_address_tokens"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# RECOVERED PAIRS
# ============================================================

recovered_df = results_df[
    results_df["recovered_by_new_rule"]
].copy()


print("\n" + "=" * 80)
print("PAIRS RECOVERED BY NEW RULE")
print("=" * 80)


if recovered_df.empty:

    print(
        "No additional true matches were recovered."
    )

else:

    print(
        recovered_df[
            [
                "s1_id",
                "target_id",
                "shared_rare_address_tokens"
            ]
        ]
        .sort_values(
            "shared_rare_address_tokens",
            ascending=False
        )
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# CHECK THE 9 PREVIOUSLY MISSED MATCHES
# ============================================================

previously_missed = [
    ("S1-867998778", "S2-126598464"),
    ("S1-867998778", "S2-648058432"),
    ("S1-867998778", "S3-807085228"),
    ("S1-42246345", "S2-112471943"),
    ("S1-42246345", "S2-819580568"),
    ("S1-174146241", "S2-906926745"),
    ("S1-216733182", "S2-860721008"),
    ("S1-97176033", "S3-581985190"),
    ("S1-983540069", "S2-4190133"),
]


print("\n" + "=" * 80)
print("CHECK OF THE 9 PREVIOUSLY MISSED MATCHES")
print("=" * 80)


for s1_id, target_id in previously_missed:

    row = results_df[
        (
            results_df["s1_id"] == s1_id
        )
        &
        (
            results_df["target_id"] == target_id
        )
    ]

    if row.empty:

        print(
            f"{s1_id} -> {target_id}: "
            "not found in sample"
        )

        continue

    row = row.iloc[0]

    print(
        f"{s1_id} -> {target_id}"
        f" | current={bool(row['current_v2'])}"
        f" | new_rule={bool(row['new_address_rule'])}"
        f" | shared_rare_address="
        f"{row['shared_rare_address_tokens']}"
    )


# ============================================================
# SAVE
# ============================================================

output_path = (
    PROJECT_ROOT
    / "output"
    / "benchmark_address_two_rare_tokens.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)

print(
    f"Results saved to:"
)

print(
    output_path
)