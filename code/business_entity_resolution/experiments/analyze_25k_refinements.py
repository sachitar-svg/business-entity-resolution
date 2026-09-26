from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "analyze_50k_recoveries.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "analyze_25k_refinements.csv"
)


df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} baseline-missed pairs")


# ---------------------------------------------------------
# The 25k rule:
# address token frequency 10,001–25,000
# AND shared number
# ---------------------------------------------------------

base_25k = (
    df["has_moderate_25k"]
    & df["has_shared_number"]
)

print("\n" + "=" * 65)
print("25K BASE RULE")
print("=" * 65)

print(
    f"25k address + shared number : "
    f"{base_25k.sum():,}"
)


# ---------------------------------------------------------
# Refinement 1:
# 25k token + shared number + 2+ moderate 25k tokens
# ---------------------------------------------------------

rule_two_25k = (
    base_25k
    & (df["moderate_25k_count"] >= 2)
)


# ---------------------------------------------------------
# Refinement 2:
# 25k token + shared number + shared name token
# ---------------------------------------------------------

rule_name = (
    base_25k
    & (df["shared_name_count"] > 0)
)


# ---------------------------------------------------------
# Refinement 3:
# 25k token + shared number + 3+ shared address tokens
# ---------------------------------------------------------

rule_address_count = (
    base_25k
    & (df["shared_address_count"] >= 3)
)


# ---------------------------------------------------------
# Refinement 4:
# 25k token + shared number + shared name token
# OR
# 2+ moderate address tokens
# ---------------------------------------------------------

rule_combined = (
    base_25k
    & (
        (df["shared_name_count"] > 0)
        | (df["moderate_25k_count"] >= 2)
    )
)


# ---------------------------------------------------------
# Refinement 5:
# 25k token + shared number + both:
# shared name AND 2+ moderate tokens
# ---------------------------------------------------------

rule_strict = (
    base_25k
    & (df["shared_name_count"] > 0)
    & (df["moderate_25k_count"] >= 2)
)


# ---------------------------------------------------------
# Add rule columns
# ---------------------------------------------------------

df["base_25k_number"] = base_25k

df["rule_two_25k"] = rule_two_25k

df["rule_name"] = rule_name

df["rule_address_count"] = rule_address_count

df["rule_combined"] = rule_combined

df["rule_strict"] = rule_strict


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("25K REFINEMENT ANALYSIS")
print("=" * 65)

print(
    f"Base 25k + shared number       : "
    f"{base_25k.sum():,}"
)

print(
    f"+ 2 moderate 25k tokens        : "
    f"{rule_two_25k.sum():,}"
)

print(
    f"+ shared name token            : "
    f"{rule_name.sum():,}"
)

print(
    f"+ >=3 shared address tokens    : "
    f"{rule_address_count.sum():,}"
)

print(
    f"+ name OR 2 moderate tokens    : "
    f"{rule_combined.sum():,}"
)

print(
    f"+ name AND 2 moderate tokens   : "
    f"{rule_strict.sum():,}"
)


# ---------------------------------------------------------
# Show the 25k base cases
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("25K + NUMBER CASES")
print("=" * 65)

base_cases = df[base_25k].copy()

if base_cases.empty:

    print("No cases found.")

else:

    print(
        base_cases[
            [
                "s1_id",
                "target_id",
                "shared_name_count",
                "shared_address_count",
                "shared_numbers_count",
                "moderate_25k_count",
                "moderate_25k_tokens"
            ]
        ].to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)
# ---------------------------------------------------------
# New combined refinement:
#
# 25k address token
# + shared number
# + (shared name OR >=3 shared address tokens)
# ---------------------------------------------------------

rule_name_or_address = (
    base_25k
    & (
        (df["shared_name_count"] > 0)
        | (df["shared_address_count"] >= 3)
    )
)

df["rule_name_or_address"] = rule_name_or_address

print(
    f"+ shared name OR >=3 address tokens : "
    f"{rule_name_or_address.sum():,}"
)