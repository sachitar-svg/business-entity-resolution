import pandas as pd

# -----------------------------
# Load training data
# -----------------------------

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

# -----------------------------
# Basic dataset sizes
# -----------------------------

print("\n===== DATASET SIZES =====")
print("Source 1:", s1.shape)
print("Source 2:", s2.shape)
print("Source 3:", s3.shape)
print("Ground Truth:", gt.shape)


# -----------------------------
# Column information
# -----------------------------

print("\n===== COLUMNS =====")
print("Source 1:", s1.columns.tolist())
print("Source 2:", s2.columns.tolist())
print("Source 3:", s3.columns.tolist())
print("Ground Truth:", gt.columns.tolist())


# -----------------------------
# Missing values
# -----------------------------

print("\n===== MISSING VALUES =====")

print("\nSource 1:")
print(s1.isnull().sum())

print("\nSource 2:")
print(s2.isnull().sum())

print("\nSource 3:")
print(s3.isnull().sum())


# -----------------------------
# Countries
# -----------------------------

print("\n===== COUNTRIES =====")

print("\nSource 1:")
print(s1["country"].value_counts(dropna=False))

print("\nSource 2:")
print(s2["country"].value_counts(dropna=False))

print("\nSource 3:")
print(s3["country"].value_counts(dropna=False))


# -----------------------------
# Ground truth examples
# -----------------------------

print("\n===== GROUND TRUTH SAMPLE =====")
print(gt.head(10).to_string(index=False))


# -----------------------------
# Number of matches per S1
# -----------------------------

def count_matches(value):
    if pd.isna(value) or str(value).strip() == "":
        return 0

    return len(str(value).split(","))


gt["match_count"] = gt["matched_entity_ids"].apply(count_matches)

print("\n===== MATCH COUNT DISTRIBUTION =====")
print(gt["match_count"].describe())

print("\nNumber of S1 entities with NO match:")
print((gt["match_count"] == 0).sum())

print("\nNumber of S1 entities with at least one match:")
print((gt["match_count"] > 0).sum())

print("\nMaximum matches for one S1 entity:")
print(gt["match_count"].max())


# -----------------------------
# Match count distribution
# -----------------------------

print("\n===== EXACT MATCH COUNTS =====")
print(gt["match_count"].value_counts().sort_index())