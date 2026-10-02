import os
import pandas as pd
from tabicl import TabICLClassifier


# =========================================================
# PROJECT
# Adaptive Health Assessment and Decision Support System
# for Induction Motors Using Vibration Signal Processing
# and Machine Learning
# =========================================================

BASE_DIR = "machine_learning"

TEST_FILE = os.path.join(BASE_DIR, "test_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "tabicl_model.pkl")


# =========================================================
# FEATURES USED BY THE MODEL
# =========================================================

FEATURES = [
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


# =========================================================
# PROJECT TITLE
# =========================================================

print("\n========================================================")
print(" Adaptive Health Assessment and Decision Support System")
print(" for Induction Motors Using Vibration Signal Processing")
print(" and Machine Learning")
print("========================================================")


# =========================================================
# INPUT DATA
# =========================================================

print("\n==================== INPUT DATA ====================\n")

test_data = pd.read_csv(TEST_FILE)

X_test = test_data[FEATURES]
y_test = test_data["fault_type"]

print("Testing samples :", len(X_test))
print("Features used   :", len(FEATURES))
print("Target          : fault_type")


# =========================================================
# MODEL
# =========================================================

print("\n==================== MODEL ====================\n")

print("Model : TabICLv2")
print("Task  : Multiclass Fault Classification")

model = TabICLClassifier.load(MODEL_FILE)

print("Trained model loaded.")


# =========================================================
# PREDICTION
# =========================================================

print("\n==================== PREDICTION ====================\n")

predictions = model.predict(X_test)

print("Prediction completed.")


# =========================================================
# PERFORMANCE CALCULATION
# =========================================================

correct = (predictions == y_test).sum()
incorrect = len(y_test) - correct

accuracy = correct / len(y_test) * 100


# =========================================================
# RESULTS
# =========================================================

print("\n==================== RESULTS ====================\n")

print("Total test samples   :", len(y_test))
print("Correct predictions  :", correct)
print("Incorrect predictions:", incorrect)

print(f"\nAccuracy : {accuracy:.2f}%")


# =========================================================
# SAMPLE PREDICTIONS
# =========================================================

print("\n==================== SAMPLE PREDICTIONS ====================\n")

for i in range(min(10, len(y_test))):
    print(
        f"Sample {i + 1}: "
        f"Actual = {y_test.iloc[i]} | "
        f"Predicted = {predictions[i]}"
    )


# =========================================================
# COMPLETE
# =========================================================

print("\n==================== PROCESS COMPLETE ====================\n")