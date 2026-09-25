from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "remaining_blocking_miss_analysis.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "analyze_50k_recoveries.csv"
)


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} missed true pairs")


# ---------------------------------------------------------
# Parse token frequencies
#
# Example:
# "central:46,256; shop:85,330; delhi:535,355"
# ---------------------------------------------------------
def parse_frequencies(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    result = []

    for item in text.split(";"):
        item = item.strip()

        if ":" not in item:
            continue

        token, freq = item.rsplit(":", 1)

        # Remove thousands separators
        freq = freq.replace(",", "").strip()

        try:
            frequency = int(float(freq))
            result.append((token.strip(), frequency))
        except ValueError:
            continue

    return result


# ---------------------------------------------------------
# Analyze each pair
# ---------------------------------------------------------
records = []

for _, row in df.iterrows():

    address_frequencies = parse_frequencies(
        row["shared_address_frequencies"]
    )

    name_frequencies = parse_frequencies(
        row["shared_name_frequencies"]
    )

    # Address tokens with frequency >10k and <=25k
    moderate_25k = [
        (token, freq)
        for token, freq in address_frequencies
        if 10_000 < freq <= 25_000
    ]

    # Address tokens with frequency >10k and <=50k
    moderate_50k = [
        (token, freq)
        for token, freq in address_frequencies
        if 10_000 < freq <= 50_000
    ]

    shared_name_count = int(row["shared_name_count"])
    shared_address_count = int(row["shared_address_count"])
    shared_numbers_count = int(row["shared_numbers_count"])

    try:
        min_name_frequency = float(row["min_shared_name_frequency"])
    except (TypeError, ValueError):
        min_name_frequency = 0

    try:
        min_address_frequency = float(row["min_shared_address_frequency"])
    except (TypeError, ValueError):
        min_address_frequency = 0

    records.append(
        {
            "s1_id": row["s1_id"],
            "target_id": row["target_id"],
            "country": row["country"],

            "s1_name": row["s1_name"],
            "target_name": row["target_name"],

            "shared_name_count": shared_name_count,
            "shared_address_count": shared_address_count,
            "shared_numbers_count": shared_numbers_count,

            "min_shared_name_frequency": min_name_frequency,
            "min_shared_address_frequency": min_address_frequency,

            "moderate_25k_count": len(moderate_25k),
            "moderate_50k_count": len(moderate_50k),

            "has_moderate_25k": len(moderate_25k) > 0,
            "has_moderate_50k": len(moderate_50k) > 0,

            "has_two_moderate_50k": len(moderate_50k) >= 2,

            "has_shared_number": shared_numbers_count > 0,

            "has_shared_name": shared_name_count > 0,

            "moderate_25k_tokens": "; ".join(
                f"{token}:{freq:,}"
                for token, freq in moderate_25k
            ),

            "moderate_50k_tokens": "; ".join(
                f"{token}:{freq:,}"
                for token, freq in moderate_50k
            ),
        }
    )


result = pd.DataFrame(records)


# ---------------------------------------------------------
# Candidate selective rules
# ---------------------------------------------------------

result["rule_25k"] = (
    result["has_moderate_25k"]
)

result["rule_50k"] = (
    result["has_moderate_50k"]
)

result["rule_50k_plus_number"] = (
    result["has_moderate_50k"]
    & result["has_shared_number"]
)

result["rule_50k_plus_two_tokens"] = (
    result["has_two_moderate_50k"]
)

result["rule_50k_plus_shared_name"] = (
    result["has_moderate_50k"]
    & result["has_shared_name"]
)


# ---------------------------------------------------------
# Save detailed analysis
# ---------------------------------------------------------
result.to_csv(OUTPUT_FILE, index=False)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("50K RECOVERY ANALYSIS")
print("=" * 60)

print(f"Total baseline-missed pairs : {len(result):,}")

print("\nRecovery counts:")
print(
    f"25k address token              : "
    f"{result['rule_25k'].sum():,}"
)

print(
    f"50k address token              : "
    f"{result['rule_50k'].sum():,}"
)

print(
    f"50k token + shared number      : "
    f"{result['rule_50k_plus_number'].sum():,}"
)

print(
    f"50k token + 2 moderate tokens  : "
    f"{result['rule_50k_plus_two_tokens'].sum():,}"
)

print(
    f"50k token + shared name        : "
    f"{result['rule_50k_plus_shared_name'].sum():,}"
)


# ---------------------------------------------------------
# Focus on 50k recovered cases
# ---------------------------------------------------------
recovered_50k = result[result["rule_50k"]].copy()

print("\n" + "=" * 60)
print("THE 50K RECOVERED PAIRS")
print("=" * 60)

print(f"Recovered by 50k              : {len(recovered_50k):,}")

print(
    f"Also recovered by 25k         : "
    f"{recovered_50k['rule_25k'].sum():,}"
)

print(
    f"50k but NOT 25k               : "
    f"{(~recovered_50k['rule_25k']).sum():,}"
)

print(
    f"Have shared number             : "
    f"{recovered_50k['has_shared_number'].sum():,}"
)

print(
    f"Have 2+ moderate address tokens: "
    f"{recovered_50k['has_two_moderate_50k'].sum():,}"
)

print(
    f"Have shared name token         : "
    f"{recovered_50k['has_shared_name'].sum():,}"
)


# ---------------------------------------------------------
# Show recovered cases
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("RECOVERED PAIRS")
print("=" * 60)

if len(recovered_50k) > 0:
    print(
        recovered_50k[
            [
                "s1_id",
                "target_id",
                "shared_name_count",
                "shared_address_count",
                "shared_numbers_count",
                "moderate_50k_count",
                "moderate_50k_tokens",
            ]
        ].to_string(index=False)
    )
else:
    print("No 50k recovered pairs found.")


print("\nSaved:")
print(OUTPUT_FILE)