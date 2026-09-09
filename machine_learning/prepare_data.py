import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = "dataset/complete_features.csv"

df = pd.read_csv(file_path)

print("\n========== ORIGINAL DATASET ==========\n")
print("Total rows:", len(df))
print("Total recordings:", df["file"].nunique())


# ==========================================
# 2. SELECT FEATURES
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
# 3. CHECK FEATURES
# ==========================================

print("\n========== FEATURES USED ==========\n")

for feature in features:
    print(feature)

print("\nTarget:", target)


# ==========================================
# 4. CHECK MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========\n")

print(df[features].isnull().sum())


# ==========================================
# 5. DEFINE RECORDING-LEVEL SPLIT
# ==========================================
#
# We split ORIGINAL RECORDINGS rather than
# individual segments to prevent data leakage.
#
# Each fault class has:
#   3 recordings -> Training
#   1 recording  -> Testing
#
# Classes:
# Normal       : 97, 98, 99, 100
# Inner Race   : 105, 106, 107, 108
# Ball         : 118, 119, 120, 121
# Outer Race   : 130, 131, 132, 133
#

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


print("\n========== TRAIN RECORDINGS ==========\n")

for file in train_files:
    print(file)

print("\n========== TEST RECORDINGS ==========\n")

for file in test_files:
    print(file)


# ==========================================
# 6. CREATE TRAINING AND TESTING DATA
# ==========================================

train_df = df[df["file"].isin(train_files)].copy()
test_df = df[df["file"].isin(test_files)].copy()


# ==========================================
# 7. CREATE X AND y
# ==========================================

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


# ==========================================
# 8. DISPLAY DATASET SIZES
# ==========================================

print("\n========== TRAINING DATA ==========\n")

print("Rows:", len(X_train))
print("Features:", X_train.shape[1])


print("\n========== TESTING DATA ==========\n")

print("Rows:", len(X_test))
print("Features:", X_test.shape[1])


# ==========================================
# 9. CLASS DISTRIBUTION
# ==========================================

print("\n========== TRAINING CLASS DISTRIBUTION ==========\n")

print(y_train.value_counts())


print("\n========== TESTING CLASS DISTRIBUTION ==========\n")

print(y_test.value_counts())


# ==========================================
# 10. RECORDING DISTRIBUTION
# ==========================================

print("\n========== TRAINING RECORDINGS BY CLASS ==========\n")

print(
    train_df.groupby("fault_type")["file"]
    .unique()
)


print("\n========== TESTING RECORDINGS BY CLASS ==========\n")

print(
    test_df.groupby("fault_type")["file"]
    .unique()
)


# ==========================================
# 11. CHECK FOR DATA LEAKAGE
# ==========================================

overlap = set(train_files).intersection(set(test_files))

print("\n========== DATA LEAKAGE CHECK ==========\n")

if len(overlap) == 0:
    print("SUCCESS: No recording appears in both training and testing data.")
else:
    print("WARNING: Data leakage detected!")
    print("Overlapping recordings:", overlap)


# ==========================================
# 12. CHECK THAT ALL CLASSES EXIST
# ==========================================

train_classes = set(y_train.unique())
test_classes = set(y_test.unique())

print("\n========== CLASS COVERAGE CHECK ==========\n")

print("Training classes:", train_classes)
print("Testing classes:", test_classes)

if train_classes == test_classes:
    print("SUCCESS: All fault classes are present in both datasets.")
else:
    print("WARNING: Training and testing classes are different.")


# ==========================================
# 13. SAVE PREPARED DATA
# ==========================================

train_output = "machine_learning/train_data.csv"
test_output = "machine_learning/test_data.csv"

train_df.to_csv(train_output, index=False)
test_df.to_csv(test_output, index=False)


print("\n========== FILES SAVED ==========\n")

print("Training data:", train_output)
print("Testing data:", test_output)


# ==========================================
# 14. FINAL SUMMARY
# ==========================================

print("\n========== FINAL SUMMARY ==========\n")

print("Original rows:", len(df))
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Number of ML features:", len(features))
print("Number of fault classes:", len(train_classes))

print("\n========== DATA PREPARATION COMPLETE ==========\n")