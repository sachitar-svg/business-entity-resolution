import pandas as pd


# ---------------------------------------
# Load ground truth
# ---------------------------------------

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

# Take the first 5 Source 1 entities
sample_gt = gt.head(5)


# ---------------------------------------
# Collect the IDs we want to inspect
# ---------------------------------------

s1_ids = set(sample_gt["source1_entity_id"])

s2_ids = set()
s3_ids = set()

for value in sample_gt["matched_entity_ids"]:
    if pd.isna(value) or str(value).strip() == "":
        continue

    for entity_id in str(value).split(","):
        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


# ---------------------------------------
# Find the required records
# without loading everything at once
# ---------------------------------------

def find_records(path, required_ids, chunk_size=500_000):

    results = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=chunk_size
    ):

        matches = chunk[chunk["entity_id"].isin(required_ids)]

        if not matches.empty:
            results.append(matches)

    if results:
        return pd.concat(results, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )


s1 = find_records(
    "dataset/train/train_source1.tsv",
    s1_ids
)

s2 = find_records(
    "dataset/train/train_source2.tsv",
    s2_ids
)

s3 = find_records(
    "dataset/train/train_source3.tsv",
    s3_ids
)


# ---------------------------------------
# Display the examples
# ---------------------------------------

print("\n==============================")
print("SOURCE 1 RECORDS")
print("==============================")

print(
    s1.to_string(index=False)
)


print("\n==============================")
print("SOURCE 2 MATCHING RECORDS")
print("==============================")

print(
    s2.to_string(index=False)
)


print("\n==============================")
print("SOURCE 3 MATCHING RECORDS")
print("==============================")

print(
    s3.to_string(index=False)
)


print("\n==============================")
print("GROUND TRUTH")
print("==============================")

print(
    sample_gt.to_string(index=False)
)