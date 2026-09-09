import pandas as pd
from tabicl import TabICLClassifier


# ============================================================
# FILE PATHS
# ============================================================

model_file = "machine_learning/models/tabicl_model.pkl"
test_file = "machine_learning/test_data.csv"


# ============================================================
# LOAD TABICLv2 MODEL
# ============================================================

print("\n========== LOADING MODEL ==========\n")

model = TabICLClassifier.load(model_file)
test_df = pd.read_csv(test_file)

print("TabICLv2 model loaded successfully.")
print("Test samples:", len(test_df))


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
# PREPARE INPUT DATA
# ============================================================

X_test = test_df[features]

print("\n========== INPUT DATA ==========\n")
print("Number of samples:", len(X_test))
print("Number of features:", len(features))


# ============================================================
# FAULT PREDICTION
# ============================================================

print("\n========== FAULT PREDICTION ==========\n")

predicted_fault = model.predict(X_test)

print("Fault prediction completed.")


# ============================================================
# PREDICTION CONFIDENCE
# ============================================================

print("\n========== PREDICTION CONFIDENCE ==========\n")

probabilities = model.predict_proba(X_test)

confidence = probabilities.max(axis=1)

print("Confidence calculation completed.")


# ============================================================
# HEALTH SCORE
# ============================================================

def calculate_health_score(fault, confidence):

    if fault == "Normal":

        score = 90 + (confidence * 10)

    elif fault == "Ball":

        score = 60 + (confidence * 20)

    elif fault == "Inner Race":

        score = 30 + (1 - confidence) * 30

    elif fault == "Outer Race":

        score = 30 + (1 - confidence) * 30

    else:

        score = 50

    return round(min(max(score, 0), 100), 2)


# ============================================================
# RISK LEVEL
# ============================================================

def determine_risk(score):

    if score >= 80:

        return "Low"

    elif score >= 60:

        return "Medium"

    elif score >= 30:

        return "High"

    else:

        return "Critical"


# ============================================================
# MAINTENANCE RECOMMENDATION
# ============================================================

def maintenance_recommendation(fault):

    if fault == "Normal":

        return "Continue normal operation and periodic monitoring."

    elif fault == "Ball":

        return "Inspect bearing condition and schedule maintenance."

    elif fault == "Inner Race":

        return "Inspect inner race bearing condition and schedule maintenance."

    elif fault == "Outer Race":

        return "Inspect outer race bearing condition and schedule maintenance."

    else:

        return "Perform detailed motor inspection."


# ============================================================
# GENERATE HEALTH ASSESSMENT
# ============================================================

health_scores = []
risk_levels = []
recommendations = []

for fault, conf in zip(predicted_fault, confidence):

    score = calculate_health_score(
        fault,
        conf
    )

    risk = determine_risk(
        score
    )

    recommendation = maintenance_recommendation(
        fault
    )

    health_scores.append(score)
    risk_levels.append(risk)
    recommendations.append(recommendation)


# ============================================================
# CREATE RESULT TABLE
# ============================================================

results = test_df[
    [
        "file",
        "segment",
        "fault_type"
    ]
].copy()

results["predicted_fault"] = predicted_fault

results["confidence"] = confidence

results["health_score"] = health_scores

results["risk_level"] = risk_levels

results["maintenance_recommendation"] = recommendations


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========== HEALTH ASSESSMENT RESULTS ==========\n")

print(
    results[
        [
            "file",
            "segment",
            "fault_type",
            "predicted_fault",
            "confidence",
            "health_score",
            "risk_level"
        ]
    ].head(20)
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

print("\n========== RISK DISTRIBUTION ==========\n")

print(
    results["risk_level"].value_counts()
)


# ============================================================
# AVERAGE HEALTH SCORE
# ============================================================

print("\n========== AVERAGE HEALTH SCORE ==========\n")

average_score = results["health_score"].mean()

print(
    "Average Health Score:",
    round(average_score, 2)
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = (
    "health_assessment/health_assessment_results.csv"
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

print("\n========== HEALTH ASSESSMENT COMPLETE ==========\n")