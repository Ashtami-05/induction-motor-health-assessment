import pandas as pd
from tabicl import TabICLClassifier

from health_assessment import health_assessment


# --------------------------------------------------
# 1. Load training and testing data
# --------------------------------------------------

train = pd.read_csv("dataset/ml/train.csv")
test = pd.read_csv("dataset/ml/test.csv")


# --------------------------------------------------
# 2. Select vibration features
# --------------------------------------------------

features = [
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "mean_absolute",
    "kurtosis",
    "skewness",
    "crest_factor"
]

X_train = train[features]
X_test = test[features]

y_train = train["fault_type"]


# --------------------------------------------------
# 3. Train TabICLv2
# --------------------------------------------------

model = TabICLClassifier()

print("Training TabICLv2...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 4. Predict test samples
# --------------------------------------------------

predictions = model.predict(X_test)

print("Number of predictions:", len(predictions))


# --------------------------------------------------
# 5. Create health assessment results
# --------------------------------------------------

results = []

for i in range(len(predictions)):

    predicted_fault = predictions[i]

    health_score, risk_level, recommendation = health_assessment(
        predicted_fault
    )

    results.append({
        "sample": i + 1,
        "actual_fault": test["fault_type"].iloc[i],
        "predicted_fault": predicted_fault,
        "health_score": health_score,
        "risk_level": risk_level,
        "recommendation": recommendation
    })


# --------------------------------------------------
# 6. Convert results to DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)


# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

results_df.to_csv(
    "health_assessment/health_assessment_results.csv",
    index=False
)


print("\nHealth assessment results saved successfully.")

print("\nFirst 5 results:")
print(results_df.head())