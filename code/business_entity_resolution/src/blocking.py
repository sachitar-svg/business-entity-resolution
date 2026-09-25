import pandas as pd

from preprocessing import normalize_name, normalize_address


def name_tokens(name):
    """Return normalized name tokens."""
    normalized = normalize_name(name)

    if not normalized:
        return set()

    return set(normalized.split())


def address_tokens(address):
    """Return normalized address tokens."""
    normalized = normalize_address(address)

    if not normalized:
        return set()

    return set(normalized.split())


def generate_candidates(source1_row, source2, source3):
    """
    Generate candidates for one Source 1 record.

    First prototype:
    - same country
    - at least one shared name token OR address token
    """

    candidates = []

    s1_country = str(source1_row["country"]).casefold().strip()

    s1_name_tokens = name_tokens(source1_row["business_name"])
    s1_address_tokens = address_tokens(source1_row["business_address"])

    for _, row in pd.concat([source2, source3], ignore_index=True).iterrows():

        candidate_country = str(row["country"]).casefold().strip()

        # Country must match
        if candidate_country != s1_country:
            continue

        candidate_name_tokens = name_tokens(row["business_name"])
        candidate_address_tokens = address_tokens(row["business_address"])

        shared_name = s1_name_tokens & candidate_name_tokens
        shared_address = s1_address_tokens & candidate_address_tokens

        # Candidate if there is at least one shared name/address token
        if shared_name or shared_address:
            candidates.append(row["entity_id"])

    return candidates


if __name__ == "__main__":

    s1 = pd.read_csv(
        "dataset/train/train_source1.tsv",
        sep="\t",
        nrows=5
    )

    s2 = pd.read_csv(
        "dataset/train/train_source2.tsv",
        sep="\t",
        nrows=1000
    )

    s3 = pd.read_csv(
        "dataset/train/train_source3.tsv",
        sep="\t",
        nrows=1000
    )

    row = s1.iloc[0]

    candidates = generate_candidates(
        row,
        s2,
        s3
    )

    print("\n===== BLOCKING TEST =====")

    print("Source 1:")
    print(row["entity_id"])

    print("Business:", row["business_name"])

    print("\nCandidates found:")
    print(len(candidates))

    print(candidates[:20])