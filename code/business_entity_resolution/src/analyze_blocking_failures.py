import re

import pandas as pd

from preprocessing import normalize_name, normalize_address
from similarity import name_similarity, address_similarity


# --------------------------------------------------
# Load the same 5 Source 1 examples
# --------------------------------------------------

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=5
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# --------------------------------------------------
# Get true matching IDs
# --------------------------------------------------

target_ids = set()

s1_matches = {}

for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    gt_row = ground_truth[
        ground_truth["source1_entity_id"] == s1_id
    ]

    if gt_row.empty:
        continue

    value = gt_row.iloc[0]["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        true_matches = []
    else:
        true_matches = [
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        ]

    s1_matches[s1_id] = true_matches
    target_ids.update(true_matches)


# --------------------------------------------------
# Collect the actual matching S2/S3 records
# --------------------------------------------------

records = {}

source_paths = [
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv"
]

print("Searching for true-match records...")

for path in source_paths:

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            chunksize=200_000
        ),
        start=1
    ):

        matches = chunk[
            chunk["entity_id"].isin(target_ids)
        ]

        for _, row in matches.iterrows():
            records[row["entity_id"]] = row.to_dict()


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def tokens(value):
    if not value:
        return set()

    return set(str(value).split())


def numeric_tokens(value):
    if not value:
        return set()

    return set(
        re.findall(r"\d+", str(value))
    )


def character_ngrams(value, n=3):
    """
    Character n-grams over normalized text.
    Spaces are kept because word boundaries
    can be informative.
    """

    if not value:
        return set()

    text = value.replace(" ", "_")

    if len(text) < n:
        return {text}

    return {
        text[i:i+n]
        for i in range(len(text) - n + 1)
    }


def jaccard(set_a, set_b):

    if not set_a and not set_b:
        return 1.0

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


# --------------------------------------------------
# Analyze every true match
# --------------------------------------------------

print("\n")
print("=" * 80)
print("BLOCKING FAILURE DIAGNOSTICS")
print("=" * 80)


for _, s1_row in s1.iterrows():

    s1_id = s1_row["entity_id"]

    true_matches = s1_matches.get(
        s1_id,
        []
    )

    if not true_matches:
        continue

    s1_name = normalize_name(
        s1_row["business_name"]
    )

    s1_address = normalize_address(
        s1_row["business_address"]
    )

    s1_name_tokens = tokens(s1_name)
    s1_address_tokens = tokens(s1_address)

    print("\n")
    print("=" * 80)
    print("SOURCE 1")
    print("=" * 80)

    print("ID:", s1_id)
    print("Name:", s1_row["business_name"])
    print("Address:", s1_row["business_address"])
    print("Country:", s1_row["country"])

    for match_id in true_matches:

        candidate = records.get(match_id)

        if candidate is None:
            print("\nMISSING RECORD:", match_id)
            continue

        candidate_name = normalize_name(
            candidate["business_name"]
        )

        candidate_address = normalize_address(
            candidate["business_address"]
        )

        candidate_name_tokens = tokens(
            candidate_name
        )

        candidate_address_tokens = tokens(
            candidate_address
        )

        shared_name_tokens = (
            s1_name_tokens &
            candidate_name_tokens
        )

        shared_address_tokens = (
            s1_address_tokens &
            candidate_address_tokens
        )

        shared_name_numbers = (
            numeric_tokens(s1_address) &
            numeric_tokens(candidate_address)
        )

        name_ngrams_s1 = character_ngrams(
            s1_name
        )

        name_ngrams_candidate = character_ngrams(
            candidate_name
        )

        address_ngrams_s1 = character_ngrams(
            s1_address
        )

        address_ngrams_candidate = character_ngrams(
            candidate_address
        )

        print("\n" + "-" * 80)
        print("TRUE MATCH:", match_id)
        print("-" * 80)

        print("Candidate name:")
        print(candidate["business_name"])

        print("\nCandidate address:")
        print(candidate["business_address"])

        print("\nCountry:")
        print(candidate["country"])

        print("\nExact shared name tokens:")
        print(sorted(shared_name_tokens))

        print("\nExact shared address tokens:")
        print(sorted(shared_address_tokens))

        print("\nShared numeric address tokens:")
        print(sorted(shared_name_numbers))

        print("\nName character 3-gram Jaccard:")
        print(
            round(
                jaccard(
                    name_ngrams_s1,
                    name_ngrams_candidate
                ),
                4
            )
        )

        print("\nAddress character 3-gram Jaccard:")
        print(
            round(
                jaccard(
                    address_ngrams_s1,
                    address_ngrams_candidate
                ),
                4
            )
        )

        print("\nName similarity:")
        print(
            round(
                name_similarity(
                    s1_row["business_name"],
                    candidate["business_name"]
                ),
                2
            )
        )

        print("\nAddress similarity:")
        print(
            round(
                address_similarity(
                    s1_row["business_address"],
                    candidate["business_address"]
                ),
                2
            )
        )