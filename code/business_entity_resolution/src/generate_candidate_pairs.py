import os
import sys
from collections import defaultdict

import pandas as pd


# ============================================================
# Make imports work when running this file directly
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocessing import normalize_name, normalize_address


# ============================================================
# CONFIGURATION
# ============================================================

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"

OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

CHUNK_SIZE = 200_000


# ============================================================
# TEXT HELPERS
# ============================================================

def safe_text(value):
    """
    Convert a value to clean text.
    """

    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    return str(value).strip()


def get_name_tokens(value):
    """
    Normalize a business name and return tokens.
    """

    normalized = normalize_name(value)

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= 2
    }


def get_address_tokens(value):
    """
    Normalize a business address and return tokens.
    """

    normalized = normalize_address(value)

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= 2
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CANDIDATE PAIR GENERATION")
print("=" * 70)


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t"
)

print(
    f"Source 1 records: {len(s1):,}"
)


# ============================================================
# BUILD BLOCKING INDEX
# ============================================================

print("\nBuilding candidate indexes...")


# token -> set(candidate entity IDs)
name_index = defaultdict(set)
address_index = defaultdict(set)

# candidate ID -> country
candidate_country = {}


source_paths = [
    S2_PATH,
    S3_PATH
]


for source_path in source_paths:

    print(
        f"\nReading {source_path}..."
    )

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            source_path,
            sep="\t",
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"  Processing chunk {chunk_number} "
            f"({len(chunk):,} records)"
        )

        for _, row in chunk.iterrows():

            candidate_id = safe_text(
                row["entity_id"]
            )

            country = safe_text(
                row["country"]
            ).casefold()

            candidate_country[
                candidate_id
            ] = country

            # ------------------------------
            # Name tokens
            # ------------------------------

            name_tokens = get_name_tokens(
                row["business_name"]
            )

            for token in name_tokens:

                name_index[
                    token
                ].add(candidate_id)

            # ------------------------------
            # Address tokens
            # ------------------------------

            address_tokens = get_address_tokens(
                row["business_address"]
            )

            for token in address_tokens:

                address_index[
                    token
                ].add(candidate_id)


# ============================================================
# PREPARE OUTPUT
# ============================================================

print("\nPreparing candidate-pair output...")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# Remove old output if it exists
if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)


# Write header
with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as output_file:

    output_file.write(
        "s1_id\tcandidate_id\n"
    )


# ============================================================
# GENERATE CANDIDATE PAIRS
# ============================================================

print("\nGenerating candidate pairs...")

total_pairs = 0
processed_s1 = 0


# ------------------------------------------------------------
# Process Source 1 in chunks
# ------------------------------------------------------------

for s1_chunk_number, s1_chunk in enumerate(
    pd.read_csv(
        S1_PATH,
        sep="\t",
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    print(
        f"\nProcessing Source 1 chunk "
        f"{s1_chunk_number} "
        f"({len(s1_chunk):,} records)"
    )

    # Open output in append mode
    with open(
        OUTPUT_PATH,
        "a",
        encoding="utf-8"
    ) as output_file:

        for _, row in s1_chunk.iterrows():

            s1_id = safe_text(
                row["entity_id"]
            )

            s1_country = safe_text(
                row["country"]
            ).casefold()

            # ------------------------------------------
            # Get Source 1 blocking tokens
            # ------------------------------------------

            name_tokens = get_name_tokens(
                row["business_name"]
            )

            address_tokens = get_address_tokens(
                row["business_address"]
            )

            # ------------------------------------------
            # Find candidates
            # ------------------------------------------

            candidates = set()

            # Name blocking
            for token in name_tokens:

                candidates.update(
                    name_index.get(
                        token,
                        set()
                    )
                )

            # Address blocking
            for token in address_tokens:

                candidates.update(
                    address_index.get(
                        token,
                        set()
                    )
                )

            # ------------------------------------------
            # Country filter + write immediately
            # ------------------------------------------

            for candidate_id in candidates:

                if (
                    candidate_country.get(
                        candidate_id,
                        ""
                    )
                    != s1_country
                ):
                    continue

                output_file.write(
                    f"{s1_id}\t{candidate_id}\n"
                )

                total_pairs += 1

            processed_s1 += 1

            # ------------------------------------------
            # Progress
            # ------------------------------------------

            if processed_s1 % 10_000 == 0:

                print(
                    f"  Processed S1 records: "
                    f"{processed_s1:,} / "
                    f"{len(s1):,} | "
                    f"Pairs written: "
                    f"{total_pairs:,}"
                )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE PAIR GENERATION COMPLETE")
print("=" * 70)

print(
    f"\nSource 1 records: "
    f"{len(s1):,}"
)

print(
    f"Candidate pairs written: "
    f"{total_pairs:,}"
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