import json
import os


NAME_STATS = (
    "code/business_entity_resolution/data/token_stats/"
    "name_token_counts.json"
)

ADDRESS_STATS = (
    "code/business_entity_resolution/data/token_stats/"
    "address_token_counts.json"
)


THRESHOLDS = [
    100,
    500,
    1_000,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
]


def load_counts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze(counts, label):
    frequencies = list(counts.values())

    print(f"\n===== {label} =====")
    print(f"Total unique tokens: {len(frequencies):,}")

    frequencies.sort()

    positions = [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99
    ]

    print("\nFrequency percentiles:")

    for p in positions:
        index = int(p * len(frequencies))
        index = min(index, len(frequencies) - 1)

        print(
            f"  {int(p * 100):>2}% of tokens have frequency <= "
            f"{frequencies[index]:,}"
        )

    print("\nTokens under frequency threshold:")

    for threshold in THRESHOLDS:
        selected = sum(
            1 for count in frequencies
            if count <= threshold
        )

        print(
            f"  <= {threshold:>7,}: "
            f"{selected:>9,} tokens "
            f"({selected / len(frequencies):.2%})"
        )

    print("\nMost rare tokens:")

    rare_tokens = sorted(
        counts.items(),
        key=lambda item: item[1]
    )[:30]

    for token, frequency in rare_tokens:
        print(f"  {token!r:35s} {frequency:,}")


if __name__ == "__main__":

    if not os.path.isfile(NAME_STATS):
        raise FileNotFoundError(NAME_STATS)

    if not os.path.isfile(ADDRESS_STATS):
        raise FileNotFoundError(ADDRESS_STATS)

    name_counts = load_counts(NAME_STATS)
    address_counts = load_counts(ADDRESS_STATS)

    analyze(name_counts, "NAME TOKENS")
    analyze(address_counts, "ADDRESS TOKENS")