import json
import os
from collections import Counter

import pandas as pd

from preprocessing import normalize_name, normalize_address


CHUNK_SIZE = 200_000

SOURCE_FILES = [
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
]

OUTPUT_DIR = "code/business_entity_resolution/data/token_stats"


def update_token_counter(counter, values, normalizer):
    """
    Add unique tokens from each record to the counter.

    A token is counted once per record, even if it appears
    multiple times inside the same record.
    """

    for value in values:

        normalized = normalizer(value)

        if not normalized:
            continue

        tokens = set(normalized.split())

        # Ignore one-character tokens for blocking statistics.
        tokens = {
            token
            for token in tokens
            if len(token) >= 2
        }

        counter.update(tokens)


def build_stats():

    name_counter = Counter()
    address_counter = Counter()

    for path in SOURCE_FILES:

        print(f"\nProcessing: {path}")

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                path,
                sep="\t",
                chunksize=CHUNK_SIZE
            ),
            start=1
        ):

            print(f"  Chunk {chunk_number}")

            update_token_counter(
                name_counter,
                chunk["business_name"],
                normalize_name
            )

            update_token_counter(
                address_counter,
                chunk["business_address"],
                normalize_address
            )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    name_path = os.path.join(
        OUTPUT_DIR,
        "name_token_counts.json"
    )

    address_path = os.path.join(
        OUTPUT_DIR,
        "address_token_counts.json"
    )

    with open(name_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(name_counter),
            f,
            ensure_ascii=False
        )

    with open(address_path, "w", encoding="utf-8") as f:
        json.dump(
            dict(address_counter),
            f,
            ensure_ascii=False
        )

    print("\n===== COMPLETE =====")

    print(
        f"Unique name tokens: {len(name_counter):,}"
    )

    print(
        f"Unique address tokens: {len(address_counter):,}"
    )

    print(f"\nSaved:")
    print(name_path)
    print(address_path)


if __name__ == "__main__":
    build_stats()