import pandas as pd

# ==========================================
# LOAD FEATURE DATASET
# ==========================================

file_path = "dataset/complete_features.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED ==========\n")
print("Total samples:", len(df))
print("Total recordings:", df["file"].nunique())


# ==========================================
# FEATURES USED FOR MACHINE LEARNING
# ==========================================

features = [
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "mean_absolute",
    "kurtosis",
    "skewness",
    "crest_factor",
    "peak_frequency",
    "peak_amplitude",
    "spectral_energy",
    "spectral_centroid",
    "spectral_bandwidth"
]

target = "fault_type"


# ==========================================
# CHECK FEATURES
# ==========================================

print("\n========== FEATURES USED ==========\n")

for feature in features:
    print("-", feature)

print("\nTotal features:", len(features))
print("Target:", target)


# ==========================================
# CHECK MISSING VALUES
# ==========================================

print("\n========== MISSING VALUE CHECK ==========\n")

print(df[features].isnull().sum())


# ==========================================
# RECORDING-LEVEL TRAIN / TEST SPLIT
# ==========================================

train_files = [
    "97.mat",
    "98.mat",
    "99.mat",
    "105.mat",
    "106.mat",
    "107.mat",
    "118.mat",
    "119.mat",
    "120.mat",
    "130.mat",
    "131.mat",
    "132.mat"
]

test_files = [
    "100.mat",
    "108.mat",
    "121.mat",
    "133.mat"
]


# ==========================================
# CREATE TRAINING AND TESTING DATA
# ==========================================

train_df = df[df["file"].isin(train_files)].copy()
test_df = df[df["file"].isin(test_files)].copy()

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


# ==========================================
# DISPLAY DATASET INFORMATION
# ==========================================

print("\n========== TRAINING DATA ==========\n")

print("Training samples:", len(X_train))
print("Training features:", X_train.shape[1])

print("\nTraining classes:")
print(y_train.value_counts())


print("\n========== TESTING DATA ==========\n")

print("Testing samples:", len(X_test))
print("Testing features:", X_test.shape[1])

print("\nTesting classes:")
print(y_test.value_counts())


# ==========================================
# CHECK RECORDING OVERLAP
# ==========================================

overlap = set(train_files).intersection(set(test_files))

print("\n========== DATA LEAKAGE CHECK ==========\n")

if len(overlap) == 0:
    print("SUCCESS: No recording appears in both training and testing data.")
else:
    print("WARNING: Recording overlap detected!")
    print(overlap)


# ==========================================
# CHECK CLASS COVERAGE
# ==========================================

train_classes = set(y_train.unique())
test_classes = set(y_test.unique())

print("\n========== CLASS COVERAGE CHECK ==========\n")

print("Training classes:", train_classes)
print("Testing classes:", test_classes)

if train_classes == test_classes:
    print("SUCCESS: All four classes are present in both datasets.")
else:
    print("WARNING: Class mismatch detected.")


# ==========================================
# SAVE TRAINING AND TESTING DATA
# ==========================================

train_output = "machine_learning/train_data.csv"
test_output = "machine_learning/test_data.csv"

train_df.to_csv(train_output, index=False)
test_df.to_csv(test_output, index=False)


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========== FILES SAVED ==========\n")

print("Training data:", train_output)
print("Testing data:", test_output)

print("\n========== FINAL SUMMARY ==========\n")

print("Original samples:", len(df))
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))
print("Number of ML features:", len(features))
print("Number of fault classes:", len(train_classes))

print("\n========== DATA PREPARATION COMPLETE ==========\n")