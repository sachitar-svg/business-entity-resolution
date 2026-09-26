import csv
from pathlib import Path

csv.field_size_limit(2**31 - 1)

test_dir = Path(r"..\business-entity-resolution\dataset\test")
candidate_file = Path("output/candidate_pairs_test.tsv")

valid_ids = set()

for filename in ["test_source2.tsv", "test_source3.tsv"]:
    path = test_dir / filename

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            valid_ids.add(row["entity_id"])

print(f"Valid S2/S3 IDs loaded: {len(valid_ids):,}")

rows = 0
invalid_ids = []
duplicate_s1 = 0
seen_s1 = set()

with open(candidate_file, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        s1_id = row["source1_entity_id"]

        if s1_id in seen_s1:
            duplicate_s1 += 1

        seen_s1.add(s1_id)

        candidates = [
            x.strip()
            for x in row["candidate_entity_ids"].split(",")
            if x.strip()
        ]

        for candidate_id in candidates:
            if candidate_id not in valid_ids:
                invalid_ids.append((s1_id, candidate_id))

        rows += 1

print(f"Candidate rows checked: {rows:,}")
print(f"Duplicate S1 rows: {duplicate_s1:,}")
print(f"Invalid S2/S3 IDs: {len(invalid_ids):,}")

if invalid_ids:
    print("First invalid IDs:")
    for item in invalid_ids[:10]:
        print(item)
else:
    print("All candidate IDs exist in test Source 2/3.")