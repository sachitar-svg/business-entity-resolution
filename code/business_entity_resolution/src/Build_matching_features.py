import re
import difflib

import pandas as pd

from preprocessing import normalize_name, normalize_address

# ============================================================
# CONFIGURATION
# ============================================================

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"

SAMPLE_SIZE = 20
RANDOM_STATE = 42

# ============================================================
# TEXT HELPERS
# ============================================================

def safe_text(value):
    """
    Convert a value to clean lowercase text.
    """

    if value is None or pd.isna(value):
        return ""

    return str(value).strip()

def get_tokens(value, normalizer):
    """
    Normalize text and return unique tokens.
    """

    normalized = normalizer(value)

    if not normalized:
        return set()

    return {
        token
        for token in normalized.split()
        if len(token) >= 2
    }

def compact_text(value):
    """
    Remove spaces and punctuation after name normalization.

    Example:
        "Prime Money"
        -> "primemoney"
    """

    normalized = normalize_name(value)

    return "".join(
        char
        for char in normalized
        if char.isalnum()
    )

def normalized_numbers(value):
    """
    Extract numbers from an address.

    Leading zeroes are removed so that:

        00123 -> 123
    """

    if value is None or pd.isna(value):
        return set()

    numbers = re.findall(
        r"\d+",
        str(value)
    )

    result = set()

    for number in numbers:

        number = number.lstrip("0")

        if number == "":
            number = "0"

        result.add(number)

    return result

# ============================================================
# SIMILARITY FUNCTIONS
# ============================================================

def jaccard_similarity(tokens_a, tokens_b):
    """
    Jaccard similarity between two token sets.
    """

    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(
        tokens_a & tokens_b
    )

    union = len(
        tokens_a | tokens_b
    )

    if union == 0:
        return 0.0

    return intersection / union

def sequence_similarity(text_a, text_b):
    """
    Character-level similarity using SequenceMatcher.
    """

    if not text_a or not text_b:
        return 0.0

    return difflib.SequenceMatcher(
        None,
        text_a,
        text_b
    ).ratio()

# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_features(
    s1_row,
    candidate_row
):
    """
    Calculate matching features for one
    Source 1 / candidate pair.
    """

    # --------------------------------------------------------
    # Raw values
    # --------------------------------------------------------

    s1_name = safe_text(
        s1_row["business_name"]
    )

    candidate_name = safe_text(
        candidate_row["business_name"]
    )

    s1_address = safe_text(
        s1_row["business_address"]
    )

    candidate_address = safe_text(
        candidate_row["business_address"]
    )

    s1_country = safe_text(
        s1_row["country"]
    ).casefold()

    candidate_country = safe_text(
        candidate_row["country"]
    ).casefold()

    # --------------------------------------------------------
    # Normalized names
    # --------------------------------------------------------

    s1_normalized_name = normalize_name(
        s1_name
    )

    candidate_normalized_name = normalize_name(
        candidate_name
    )

    s1_compact_name = compact_text(
        s1_name
    )

    candidate_compact_name = compact_text(
        candidate_name
    )

    # --------------------------------------------------------
    # Name tokens
    # --------------------------------------------------------

    s1_name_tokens = get_tokens(
        s1_name,
        normalize_name
    )

    candidate_name_tokens = get_tokens(
        candidate_name,
        normalize_name
    )

    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    s1_address_tokens = get_tokens(
        s1_address,
        normalize_address
    )

    candidate_address_tokens = get_tokens(
        candidate_address,
        normalize_address
    )

    # --------------------------------------------------------
    # Address numbers
    # --------------------------------------------------------

    s1_numbers = normalized_numbers(
        s1_address
    )

    candidate_numbers = normalized_numbers(
        candidate_address
    )

    # ========================================================
    # NAME FEATURES
    # ========================================================

    name_exact = int(
        bool(
            s1_normalized_name
            and candidate_normalized_name
            and
            s1_normalized_name
            == candidate_normalized_name
        )
    )

    name_compact_exact = int(
        bool(
            s1_compact_name
            and candidate_compact_name
            and
            s1_compact_name
            == candidate_compact_name
        )
    )

    name_token_jaccard = (
        jaccard_similarity(
            s1_name_tokens,
            candidate_name_tokens
        )
    )

    name_char_similarity = (
        sequence_similarity(
            s1_normalized_name,
            candidate_normalized_name
        )
    )

    # ========================================================
    # ADDRESS FEATURES
    # ========================================================

    address_token_jaccard = (
        jaccard_similarity(
            s1_address_tokens,
            candidate_address_tokens
        )
    )

    address_char_similarity = (
        sequence_similarity(
            normalize_address(s1_address),
            normalize_address(candidate_address)
        )
    )

    # ========================================================
    # ADDRESS NUMBER FEATURES
    # ========================================================

    number_intersection = (
        s1_numbers
        &
        candidate_numbers
    )

    number_union = (
        s1_numbers
        |
        candidate_numbers
    )

    number_overlap = int(
        len(number_intersection) > 0
    )

    number_jaccard = (
        len(number_intersection)
        /
        len(number_union)
        if number_union
        else 0.0
    )

    # ========================================================
    # COUNTRY FEATURE
    # ========================================================

    country_match = int(
        bool(
            s1_country
            and candidate_country
            and
            s1_country
            == candidate_country
        )
    )

    # ========================================================
    # MISSING VALUE FEATURES
    # ========================================================

    s1_name_missing = int(
        not bool(s1_name)
    )

    candidate_name_missing = int(
        not bool(candidate_name)
    )

    s1_address_missing = int(
        not bool(s1_address)
    )

    candidate_address_missing = int(
        not bool(candidate_address)
    )

    # ========================================================
    # RETURN FEATURE DICTIONARY
    # ========================================================

    return {

        # IDs
        "s1_id": s1_row["entity_id"],
        "candidate_id": candidate_row["entity_id"],

        # Name features
        "name_exact": name_exact,
        "name_compact_exact": name_compact_exact,
        "name_token_jaccard": name_token_jaccard,
        "name_char_similarity": name_char_similarity,

        # Address features
        "address_token_jaccard": address_token_jaccard,
        "address_char_similarity": address_char_similarity,

        # Number features
        "number_overlap": number_overlap,
        "number_jaccard": number_jaccard,

        # Country
        "country_match": country_match,

        # Missing values
        "s1_name_missing": s1_name_missing,
        "candidate_name_missing": candidate_name_missing,
        "s1_address_missing": s1_address_missing,
        "candidate_address_missing": candidate_address_missing,
    }

if __name__ == "__main__":
    # ============================================================
    # LOAD DATA
    # ============================================================

    print("=" * 70)
    print("MATCHING FEATURE TEST")
    print("=" * 70)

    print("\nLoading Source 1...")

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t"
    )

    print(
        f"Source 1 records: {len(s1):,}"
    )

    print("\nLoading Source 2...")

    s2 = pd.read_csv(
        S2_PATH,
        sep="\t"
    )

    print(
        f"Source 2 records: {len(s2):,}"
    )

    # ============================================================
    # SMALL TEST SAMPLE
    # ============================================================

    print(
        f"\nSelecting {SAMPLE_SIZE} Source 1 records..."
    )

    s1_sample = s1.sample(
        n=min(
            SAMPLE_SIZE,
            len(s1)
        ),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # ============================================================
    # TEMPORARY TEST
    #
    # IMPORTANT:
    # We are NOT doing full candidate generation yet.
    #
    # This simply tests the feature calculation against
    # a small number of Source 2 records.
    # ============================================================

    s2_sample = s2.head(50).copy()

    # ============================================================
    # CALCULATE FEATURES
    # ============================================================

    feature_rows = []

    print("\nCalculating test features...")

    for _, s1_row in s1_sample.iterrows():

        for _, candidate_row in s2_sample.iterrows():

            features = calculate_features(
                s1_row,
                candidate_row
            )

            feature_rows.append(
                features
            )

    features_df = pd.DataFrame(
        feature_rows
    )

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================

    print("\n" + "=" * 70)
    print("FEATURE TEST RESULTS")
    print("=" * 70)

    print(
        f"\nFeature rows generated: "
        f"{len(features_df):,}"
    )

    print(
        f"Feature columns: "
        f"{len(features_df.columns):,}"
    )

    print("\nColumns:")

    for column in features_df.columns:

        print(
            f"  - {column}"
        )

    print("\nFirst 10 rows:")

    print(
        features_df
        .head(10)
        .to_string(index=False)
    )

    # ============================================================
    # BASIC STATISTICS
    # ============================================================

    print("\n" + "=" * 70)
    print("FEATURE STATISTICS")
    print("=" * 70)

    numeric_columns = [
        column
        for column in features_df.columns
        if column not in {
            "s1_id",
            "candidate_id"
        }
    ]

    print(
        features_df[
            numeric_columns
        ].describe()
        .T
        .to_string()
    )

    print("\nDone.")
