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

print("Test samples :", len(test_data))


# =========================================================
# FEATURES USED BY TABICLv2
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

X_test = test_data[FEATURES]

print("Features used:", len(FEATURES))


# =========================================================
# MODEL
# =========================================================

print("\n==================== MODEL ====================\n")

print("Model : TabICLv2")
print("Task  : Multiclass Fault Classification")

model = TabICLClassifier.load(MODEL_FILE)

print("Trained model loaded.")


# =========================================================
# FAULT PREDICTION
# =========================================================

print("\n==================== FAULT PREDICTION ====================\n")

predictions = model.predict(X_test)

print("Fault prediction completed.")


# =========================================================
# PREDICTION CONFIDENCE
# =========================================================

probabilities = model.predict_proba(X_test)

confidence = probabilities.max(axis=1)

print("Prediction confidence calculated.")


# =========================================================
# HEALTH ASSESSMENT
# =========================================================
#
# This is a project-specific decision-support assessment.
#
# It does NOT represent a physical motor health percentage.
#
# The model identifies the predicted fault condition.
# The system then assigns a risk level and maintenance action.
# Confidence is displayed separately.
# =========================================================

def determine_health_status(fault):

    if fault == "Normal":
        return "Healthy"

    elif fault == "Ball":
        return "Fault Detected"

    elif fault == "Inner Race":
        return "Fault Detected"

    elif fault == "Outer Race":
        return "Fault Detected"

    else:
        return "Requires Inspection"


# =========================================================
# RISK LEVEL
# =========================================================

def determine_risk(fault):

    if fault == "Normal":
        return "Low"

    elif fault == "Ball":
        return "High"

    elif fault == "Inner Race":
        return "High"

    elif fault == "Outer Race":
        return "High"

    else:
        return "Medium"


# =========================================================
# MAINTENANCE RECOMMENDATION
# =========================================================

def maintenance_recommendation(fault):

    if fault == "Normal":
        return "Continue normal operation and periodic monitoring."

    elif fault == "Ball":
        return "Inspect bearing and schedule maintenance."

    elif fault == "Inner Race":
        return "Inspect inner race and schedule maintenance."

    elif fault == "Outer Race":
        return "Inspect outer race and schedule maintenance."

    else:
        return "Perform detailed motor inspection."


# =========================================================
# CREATE ASSESSMENT RESULTS
# =========================================================

results = []

for i in range(len(predictions)):

    fault = predictions[i]
    conf = confidence[i]

    health_status = determine_health_status(fault)
    risk = determine_risk(fault)
    recommendation = maintenance_recommendation(fault)

    results.append({
        "sample": i + 1,
        "predicted_fault": fault,
        "confidence": round(conf * 100, 2),
        "health_status": health_status,
        "risk_level": risk,
        "maintenance_recommendation": recommendation
    })


results_df = pd.DataFrame(results)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n==================== HEALTH ASSESSMENT ====================\n")

print(
    results_df.head(10).to_string(index=False)
)


# =========================================================
# SUMMARY
# =========================================================

print("\n==================== ASSESSMENT SUMMARY ====================\n")

print("Health Status Distribution:")

print(
    results_df["health_status"].value_counts()
)

print("\nRisk Level Distribution:")

print(
    results_df["risk_level"].value_counts()
)


# =========================================================
# SAVE RESULTS
# =========================================================

output_directory = "health_assessment"

os.makedirs(output_directory, exist_ok=True)

output_file = os.path.join(
    output_directory,
    "health_assessment_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# =========================================================
# COMPLETE
# =========================================================

print("\n==================== PROCESS COMPLETE ====================\n")