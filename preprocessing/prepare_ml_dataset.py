import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# ML Dataset Preparation
# Project: Induction Motor Health Assessment
# ---------------------------------------------------------

INPUT_FILE = os.path.join("..", "dataset", "features.csv")
OUTPUT_DIR = os.path.join("..", "dataset", "ml")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load feature dataset
df = pd.read_csv(INPUT_FILE)

print("Original dataset:")
print(df["fault_type"].value_counts())

# ---------------------------------------------------------
# Balance the classes
# ---------------------------------------------------------

min_count = df["fault_type"].value_counts().min()

print(f"\nSamples selected per class: {min_count}")

balanced_df = (
    df.groupby("fault_type", group_keys=False)
      .sample(n=min_count, random_state=42)
      .reset_index(drop=True)
)

print("\nBalanced dataset:")
print(balanced_df["fault_type"].value_counts())

# ---------------------------------------------------------
# Split by ORIGINAL RECORDING
# ---------------------------------------------------------

# Unique CWRU recordings
files = balanced_df["file"].unique()

train_files, test_files = train_test_split(
    files,
    test_size=0.25,
    random_state=42,
    stratify=[
        balanced_df[balanced_df["file"] == f]["fault_type"].iloc[0]
        for f in files
    ]
)

train_df = balanced_df[
    balanced_df["file"].isin(train_files)
].copy()

test_df = balanced_df[
    balanced_df["file"].isin(test_files)
].copy()

# ---------------------------------------------------------
# Save datasets
# ---------------------------------------------------------

train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)

# Save balanced complete dataset
balanced_df.to_csv(
    os.path.join(OUTPUT_DIR, "balanced_dataset.csv"),
    index=False
)

# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n-----------------------------------------")
print("ML dataset preparation completed.")
print("-----------------------------------------")

print(f"Total balanced samples: {len(balanced_df)}")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

print("\nTraining class distribution:")
print(train_df["fault_type"].value_counts())

print("\nTesting class distribution:")
print(test_df["fault_type"].value_counts())

print("\nTraining recordings:")
print(sorted(train_files))

print("\nTesting recordings:")
print(sorted(test_files))

print("-----------------------------------------")
print("Files saved in: ../dataset/ml/")
print("-----------------------------------------")
