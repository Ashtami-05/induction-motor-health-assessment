import os
import pandas as pd
from tabicl import TabICLClassifier


# ============================================================
# FILE PATHS
# ============================================================

model_file = "machine_learning/models/tabicl_model.pkl"
test_file = "machine_learning/test_data.csv"

output_file = "machine_learning/results/predictions.csv"


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========== LOADING TABICLv2 MODEL ==========\n")

model = TabICLClassifier.load(model_file)

print("TabICLv2 model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n========== LOADING INPUT DATA ==========\n")

df = pd.read_csv(test_file)

X = df[features]

print("Input samples:", len(X))
print("Number of features:", len(features))


# ============================================================
# PREDICT FAULT
# ============================================================

print("\n========== FAULT PREDICTION ==========\n")

predicted_fault = model.predict(X)

print("Fault prediction completed.")


# ============================================================
# PREDICTION PROBABILITY
# ============================================================

print("\n========== PREDICTION PROBABILITY ==========\n")

probabilities = model.predict_proba(X)

confidence = probabilities.max(axis=1)

print("Prediction confidence calculated.")


# ============================================================
# CREATE OUTPUT
# ============================================================

results = df[
    [
        "file",
        "segment",
        "fault_type"
    ]
].copy()

results["predicted_fault"] = predicted_fault
results["confidence"] = confidence


# ============================================================
# CHECK PREDICTIONS
# ============================================================

correct_predictions = (
    results["fault_type"] ==
    results["predicted_fault"]
).sum()

total_predictions = len(results)

accuracy = (
    correct_predictions /
    total_predictions
) * 100


print("\n========== PREDICTION SUMMARY ==========\n")

print("Total predictions:", total_predictions)
print("Correct predictions:", correct_predictions)
print("Incorrect predictions:", total_predictions - correct_predictions)
print(f"Prediction accuracy: {accuracy:.2f}%")


# ============================================================
# DISPLAY FIRST 20 PREDICTIONS
# ============================================================

print("\n========== FIRST 20 PREDICTIONS ==========\n")

print(
    results[
        [
            "file",
            "segment",
            "fault_type",
            "predicted_fault",
            "confidence"
        ]
    ].head(20)
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "machine_learning/results",
    exist_ok=True
)

results.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========== FILE SAVED ==========\n")

print(output_file)

print("\n========== PREDICTION MODULE COMPLETE ==========\n")