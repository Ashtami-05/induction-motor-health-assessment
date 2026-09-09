import pandas as pd

# Path to Member 2's completed feature dataset
file_path = "dataset/complete_features.csv"

# Load dataset
df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========\n")

# Basic information
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\n========== COLUMN NAMES ==========\n")

for i, column in enumerate(df.columns, start=1):
    print(i, ":", column)

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== FAULT CLASSES ==========\n")
print(df["fault_type"].value_counts())

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========\n")
print("Number of duplicate rows:", df.duplicated().sum())

print("\n========== DATA TYPES ==========\n")
print(df.dtypes)

print("\n========== ORIGINAL RECORDINGS ==========\n")
print("Number of unique files:", df["file"].nunique())

print("\nFiles:")
print(df["file"].unique())
print("\n========== RPM MISSING BY FILE ==========\n")

print(
    df.groupby("file")["rpm"]
      .apply(lambda x: x.isnull().sum())
)

print("\n========== RPM VALUES BY FILE ==========\n")

print(
    df.groupby("file")["rpm"]
      .first()
)

print("\n========== DATASET CHECK COMPLETE ==========\n")