import re
from collections import Counter

import pandas as pd

from preprocessing import normalize_name, normalize_address


def get_tokens(value):
    """Return unique normalized tokens from a text field."""

    if pd.isna(value):
        return set()

    text = str(value)

    if not text:
        return set()

    return set(text.split())


def count_tokens(path, field, chunk_size=200_000):
    """
    Count how many records contain each token.

    We count a token once per record, not once per occurrence.
    """

    counter = Counter()

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            chunksize=chunk_size
        ),
        start=1
    ):

        print(
            f"Processing {path} - chunk {chunk_number}"
        )

        for value in chunk[field]:

            if field == "business_name":
                value = normalize_name(value)
            else:
                value = normalize_address(value)

            tokens = get_tokens(value)

            counter.update(tokens)

    return counter


if __name__ == "__main__":

    source_paths = [
        "dataset/train/train_source2.tsv",
        "dataset/train/train_source3.tsv"
    ]

    print("\n===== NAME TOKEN FREQUENCY =====")

    name_counter = Counter()

    for path in source_paths:
        counter = count_tokens(
            path,
            "business_name"
        )
        name_counter.update(counter)

    print("\nMost common name tokens:")
    for token, count in name_counter.most_common(30):
        print(f"{token:30s} {count:,}")

    print("\n===== ADDRESS TOKEN FREQUENCY =====")

    address_counter = Counter()

    for path in source_paths:
        counter = count_tokens(
            path,
            "business_address"
        )
        address_counter.update(counter)

    print("\nMost common address tokens:")
    for token, count in address_counter.most_common(30):
        print(f"{token:30s} {count:,}")