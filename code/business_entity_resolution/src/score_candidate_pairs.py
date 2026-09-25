import os
import sys
import math
from collections import defaultdict

import pandas as pd


# ============================================================
# IMPORT PREPROCESSING
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocessing import (
    normalize_name,
    normalize_address
)


# ============================================================
# PATHS
# ============================================================

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"

CANDIDATE_PATH = (
    "output/candidate_pairs.tsv"
)

OUTPUT_PATH = (
    "output/matched_entities.tsv"
)

CHUNK_SIZE = 100_000


# ============================================================
# SCORING WEIGHTS
# ============================================================

NAME_WEIGHT = 0.55
ADDRESS_WEIGHT = 0.35
COUNTRY_WEIGHT = 0.10


# ============================================================
# TEXT HELPERS
# ============================================================

def safe_text(value):
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def tokenize(value):
    """
    Normalize text and return a set of tokens.
    """

    normalized = value

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= 2
    }


# ============================================================
# TOKEN SIMILARITY
# ============================================================

def token_similarity(text1, text2):
    """
    Jaccard token similarity.

    Example:

        {abc, company}
        {abc, company, ltd}

    similarity = 2 / 3
    """

    if not text1 or not text2:
        return 0.0

    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(
        tokens1.intersection(tokens2)
    )

    union = len(
        tokens1.union(tokens2)
    )

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# CHARACTER SIMILARITY
# ============================================================

def character_similarity(text1, text2):
    """
    Lightweight character similarity.

    Uses common-character overlap rather than an expensive
    edit-distance calculation, so the full dataset can be
    processed much faster.
    """

    if not text1 or not text2:
        return 0.0

    if text1 == text2:
        return 1.0

    # Fast containment signal
    if (
        len(text1) >= 4
        and text1 in text2
    ):
        return 0.85

    if (
        len(text2) >= 4
        and text2 in text1
    ):
        return 0.85

    # Character bigrams
    if len(text1) < 2 or len(text2) < 2:
        return 0.0

    grams1 = {
        text1[i:i + 2]
        for i in range(len(text1) - 1)
    }

    grams2 = {
        text2[i:i + 2]
        for i in range(len(text2) - 1)
    }

    union = grams1.union(grams2)

    if not union:
        return 0.0

    intersection = grams1.intersection(
        grams2
    )

    return len(intersection) / len(union)


# ============================================================
# FIELD SIMILARITY
# ============================================================

def field_similarity(
    value1,
    value2,
    normalizer
):
    """
    Combine token and character similarity.
    """

    normalized1 = normalizer(
        value1
    )

    normalized2 = normalizer(
        value2
    )

    if not normalized1 or not normalized2:
        return 0.0

    if normalized1 == normalized2:
        return 1.0

    token_score = token_similarity(
        normalized1,
        normalized2
    )

    char_score = character_similarity(
        normalized1.replace(" ", ""),
        normalized2.replace(" ", "")
    )

    return (
        0.60 * token_score
        + 0.40 * char_score
    )


# ============================================================
# LOAD SOURCE DATA
# ============================================================

print("=" * 70)
print("CANDIDATE PAIR SCORING")
print("=" * 70)


print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(
    f"Source 1 records: "
    f"{len(s1):,}"
)


print("\nLoading Source 2...")

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(
    f"Source 2 records: "
    f"{len(s2):,}"
)


print("\nLoading Source 3...")

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(
    f"Source 3 records: "
    f"{len(s3):,}"
)


# ============================================================
# BUILD CANDIDATE LOOKUP
# ============================================================

print("\nBuilding candidate lookup...")


candidate_records = {}


# ------------------------------------------------------------
# Source 2
# ------------------------------------------------------------

for row in s2.itertuples(
    index=False
):

    entity_id = safe_text(
        row.entity_id
    )

    if not entity_id:
        continue

    candidate_records[
        entity_id
    ] = {
        "business_name": safe_text(
            row.business_name
        ),
        "business_address": safe_text(
            row.business_address
        ),
        "country": safe_text(
            row.country
        ).casefold()
    }


# ------------------------------------------------------------
# Source 3
# ------------------------------------------------------------

for row in s3.itertuples(
    index=False
):

    entity_id = safe_text(
        row.entity_id
    )

    if not entity_id:
        continue

    candidate_records[
        entity_id
    ] = {
        "business_name": safe_text(
            row.business_name
        ),
        "business_address": safe_text(
            row.business_address
        ),
        "country": safe_text(
            row.country
        ).casefold()
    }


print(
    f"Candidate records loaded: "
    f"{len(candidate_records):,}"
)


# ============================================================
# SOURCE 1 LOOKUP
# ============================================================

print("\nBuilding Source-1 lookup...")


source1_records = {}


for row in s1.itertuples(
    index=False
):

    entity_id = safe_text(
        row.entity_id
    )

    if not entity_id:
        continue

    source1_records[
        entity_id
    ] = {
        "business_name": safe_text(
            row.business_name
        ),
        "business_address": safe_text(
            row.business_address
        ),
        "country": safe_text(
            row.country
        ).casefold()
    }


# ============================================================
# PREPARE OUTPUT
# ============================================================

if os.path.exists(
    OUTPUT_PATH
):

    os.remove(
        OUTPUT_PATH
    )


os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as output_file:

    output_file.write(
        "s1_id\tcandidate_id\t"
        "score\tname_score\t"
        "address_score\tcountry_match\n"
    )


# ============================================================
# SCORE CANDIDATE PAIRS
# ============================================================

print("\n" + "=" * 70)
print("SCORING CANDIDATE PAIRS")
print("=" * 70)


best_matches = {}

processed_pairs = 0
skipped_pairs = 0


candidate_reader = pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE
)


for chunk_number, chunk in enumerate(
    candidate_reader,
    start=1
):

    print(
        f"\nProcessing candidate chunk "
        f"{chunk_number:,} "
        f"({len(chunk):,} pairs)",
        flush=True
    )


    for row in chunk.itertuples(
        index=False
    ):

        s1_id = safe_text(
            row.s1_id
        )

        candidate_id = safe_text(
            row.candidate_id
        )


        if not s1_id or not candidate_id:

            skipped_pairs += 1

            continue


        source1 = source1_records.get(
            s1_id
        )

        candidate = candidate_records.get(
            candidate_id
        )


        if source1 is None or candidate is None:

            skipped_pairs += 1

            continue


        # ====================================================
        # COUNTRY
        # ====================================================

        country_match = (
            source1["country"]
            != ""
            and
            candidate["country"]
            != ""
            and
            source1["country"]
            == candidate["country"]
        )


        country_score = (
            1.0
            if country_match
            else 0.0
        )


        # ====================================================
        # NAME
        # ====================================================

        name_score = field_similarity(
            source1["business_name"],
            candidate["business_name"],
            normalize_name
        )


        # ====================================================
        # ADDRESS
        # ====================================================

        address_score = field_similarity(
            source1["business_address"],
            candidate["business_address"],
            normalize_address
        )


        # ====================================================
        # COMBINED SCORE
        # ====================================================

        score = (
            NAME_WEIGHT * name_score
            +
            ADDRESS_WEIGHT * address_score
            +
            COUNTRY_WEIGHT * country_score
        )


        # ====================================================
        # EXACT MATCH BOOST
        # ====================================================

        normalized_s1_name = normalize_name(
            source1["business_name"]
        )

        normalized_candidate_name = normalize_name(
            candidate["business_name"]
        )

        normalized_s1_address = normalize_address(
            source1["business_address"]
        )

        normalized_candidate_address = normalize_address(
            candidate["business_address"]
        )


        exact_name = (
            normalized_s1_name != ""
            and
            normalized_s1_name
            ==
            normalized_candidate_name
        )


        exact_address = (
            normalized_s1_address != ""
            and
            normalized_s1_address
            ==
            normalized_candidate_address
        )


        if exact_name:
            score += 0.10

        if exact_address:
            score += 0.10


        # Maximum score = 1.20
        # Keep it normalized to 0-1.
        score = min(
            score,
            1.0
        )


        # ====================================================
        # KEEP BEST MATCH
        # ====================================================

        previous = best_matches.get(
            s1_id
        )


        if (
            previous is None
            or
            score > previous["score"]
        ):

            best_matches[
                s1_id
            ] = {
                "candidate_id": candidate_id,
                "score": score,
                "name_score": name_score,
                "address_score": address_score,
                "country_match": int(
                    country_match
                )
            }


        processed_pairs += 1


    print(
        f"  Pairs processed: "
        f"{processed_pairs:,}"
        f" | S1 matches stored: "
        f"{len(best_matches):,}",
        flush=True
    )


# ============================================================
# WRITE BEST MATCHES
# ============================================================

print("\nWriting best matches...")


with open(
    OUTPUT_PATH,
    "a",
    encoding="utf-8"
) as output_file:

    for s1_id, result in best_matches.items():

        output_file.write(
            f"{s1_id}\t"
            f"{result['candidate_id']}\t"
            f"{result['score']:.6f}\t"
            f"{result['name_score']:.6f}\t"
            f"{result['address_score']:.6f}\t"
            f"{result['country_match']}\n"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("SCORING COMPLETE")
print("=" * 70)

print(
    f"\nCandidate pairs processed: "
    f"{processed_pairs:,}"
)

print(
    f"Pairs skipped: "
    f"{skipped_pairs:,}"
)

print(
    f"S1 entities with best match: "
    f"{len(best_matches):,}"
)

print(
    "\nOutput saved to:"
)

print(
    os.path.abspath(
        OUTPUT_PATH
    )
)

print("\nDone.")