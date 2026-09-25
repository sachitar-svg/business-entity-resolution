import pandas as pd

from preprocessing import normalize_name, normalize_address


# ---------------------------------------
# Load only a small sample of Source 1
# ---------------------------------------

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t",
    nrows=5
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# ---------------------------------------
# Prepare information for each S1 record
# ---------------------------------------

s1_info = {}

for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    name = normalize_name(row["business_name"])
    address = normalize_address(row["business_address"])

    s1_info[s1_id] = {
        "country": str(row["country"]).casefold().strip(),
        "name_tokens": set(name.split()) if name else set(),
        "address_tokens": set(address.split()) if address else set(),
        "candidates": set()
    }


# ---------------------------------------
# Read Source 2 and Source 3 in chunks
# ---------------------------------------

source_paths = [
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv"
]


for source_path in source_paths:

    print(f"\nProcessing: {source_path}")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            source_path,
            sep="\t",
            chunksize=200_000
        )
    ):

        print(f"  Processing chunk {chunk_number + 1}")

        for _, row in chunk.iterrows():

            candidate_country = (
                str(row["country"]).casefold().strip()
            )

            candidate_name = normalize_name(
                row["business_name"]
            )

            candidate_address = normalize_address(
                row["business_address"]
            )

            candidate_name_tokens = (
                set(candidate_name.split())
                if candidate_name
                else set()
            )

            candidate_address_tokens = (
                set(candidate_address.split())
                if candidate_address
                else set()
            )

            # Compare against our 5 S1 records
            for s1_id, info in s1_info.items():

                # Country must match
                if candidate_country != info["country"]:
                    continue

                shared_name = (
                    info["name_tokens"] &
                    candidate_name_tokens
                )

                shared_address = (
                    info["address_tokens"] &
                    candidate_address_tokens
                )

                if shared_name or shared_address:

                    info["candidates"].add(
                        row["entity_id"]
                    )


# ---------------------------------------
# Compare candidates with ground truth
# ---------------------------------------

print("\n")
print("=" * 50)
print("BLOCKING EVALUATION")
print("=" * 50)


for s1_id, info in s1_info.items():

    gt_row = ground_truth[
        ground_truth["source1_entity_id"] == s1_id
    ]

    if gt_row.empty:
        print(f"\n{s1_id}: ground truth not found")
        continue

    value = gt_row.iloc[0]["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        true_matches = set()
    else:
        true_matches = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }

    candidates = info["candidates"]

    captured = true_matches & candidates
    missed = true_matches - candidates

    if true_matches:
        recall = len(captured) / len(true_matches)
    else:
        recall = 1.0 if not candidates else 0.0

    print("\n----------------------------------------")
    print("S1:", s1_id)

    print("True matches:", len(true_matches))
    print("Candidates:", len(candidates))
    print("Captured:", len(captured))
    print("Missed:", len(missed))

    print("Blocking recall:", round(recall, 4))

    if missed:
        print("\nMissed true matches:")
        for entity_id in sorted(missed):
            print(" ", entity_id)