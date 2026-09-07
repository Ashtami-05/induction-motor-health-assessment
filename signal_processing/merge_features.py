import pandas as pd

# Load the existing time-domain features
time_features = pd.read_csv("dataset/features.csv")

# Load the new frequency-domain features
frequency_features = pd.read_csv("dataset/frequency_features.csv")

print("Time-domain dataset shape:", time_features.shape)
print("Frequency-domain dataset shape:", frequency_features.shape)

# Merge using file name and segment number
complete_features = pd.merge(
    time_features,
    frequency_features,
    on=["file", "segment"],
    how="inner"
)

# Save the complete feature dataset
output_file = "dataset/complete_features.csv"
complete_features.to_csv(output_file, index=False)

print()
print("Feature merging completed.")
print("Complete dataset shape:", complete_features.shape)
print("Saved to:", output_file)
print()
print("Columns:")
print(complete_features.columns.tolist())