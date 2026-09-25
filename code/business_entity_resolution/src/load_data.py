import pandas as pd

# Paths to training files
train_s1_path = "dataset/train/train_source1.tsv"
train_s2_path = "dataset/train/train_source2.tsv"
train_s3_path = "dataset/train/train_source3.tsv"
ground_truth_path = "dataset/train/train_ground_truth.tsv"

# Load only the first 5 rows initially
train_s1 = pd.read_csv(train_s1_path, sep="\t", nrows=5)
train_s2 = pd.read_csv(train_s2_path, sep="\t", nrows=5)
train_s3 = pd.read_csv(train_s3_path, sep="\t", nrows=5)
ground_truth = pd.read_csv(ground_truth_path, sep="\t", nrows=5)

print("\nSOURCE 1")
print(train_s1)

print("\nSOURCE 2")
print(train_s2)

print("\nSOURCE 3")
print(train_s3)

print("\nGROUND TRUTH")
print(ground_truth)

print("\nCOLUMNS")
print("Source 1:", train_s1.columns.tolist())
print("Source 2:", train_s2.columns.tolist())
print("Source 3:", train_s3.columns.tolist())
print("Ground truth:", ground_truth.columns.tolist())