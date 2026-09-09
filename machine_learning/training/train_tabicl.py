import os
import pandas as pd

from tabicl import TabICLClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# FILE PATHS
# ============================================================

train_file = "machine_learning/train_data.csv"
test_file = "machine_learning/test_data.csv"

os.makedirs("machine_learning/models", exist_ok=True)
os.makedirs("machine_learning/results", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("\n========== DATA LOADED ==========\n")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

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

target = "fault_type"

X_train = train_df[features]
X_test = test_df[features]

y_train = train_df[target]
y_test = test_df[target]

print("\n========== ML DATA ==========\n")
print("Number of features:", len(features))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nFeatures used:")
for feature in features:
    print("-", feature)

# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUE CHECK ==========\n")

print("Training missing values:")
print(X_train.isnull().sum())

print("\nTesting missing values:")
print(X_test.isnull().sum())

# ============================================================
# CREATE TABICL MODEL
# ============================================================

print("\n========== TABICLv2 MODEL ==========\n")

model = TabICLClassifier(
    n_estimators=8,
    random_state=42,
    device=None,
    verbose=True
)

print("TabICLv2 model created.")

# ============================================================
# TRAIN / FIT MODEL
# ============================================================

print("\n========== TABICLv2 FITTING ==========\n")

print("Starting TabICLv2...")
print("The first run may download the pretrained checkpoint.")

model.fit(X_train, y_train)

print("\nTabICLv2 fitting completed.")

# ============================================================
# PREDICTION
# ============================================================

print("\n========== PREDICTION ==========\n")

y_pred = model.predict(X_test)

print("Prediction completed.")

# ============================================================
# PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n========== TABICLv2 PERFORMANCE ==========\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========\n")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [
    "Ball",
    "Inner Race",
    "Normal",
    "Outer Race"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\n========== CONFUSION MATRIX ==========\n")
print(cm)

confusion_matrix_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

confusion_matrix_file = (
    "machine_learning/results/tabicl_confusion_matrix.csv"
)

confusion_matrix_df.to_csv(
    confusion_matrix_file
)

# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Model": ["TabICLv2"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1]
})

metrics_file = "machine_learning/results/tabicl_metrics.csv"

metrics.to_csv(
    metrics_file,
    index=False
)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = test_df[
    ["file", "segment", "fault_type"]
].copy()

prediction_df["predicted_fault"] = y_pred

prediction_file = (
    "machine_learning/results/tabicl_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    "machine_learning/models/tabicl_model.pkl"
)

print("\n========== SAVING MODEL ==========\n")

model.save(model_file)

print("TabICLv2 model saved.")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== FILES SAVED ==========\n")

print("Model:")
print(model_file)

print("\nMetrics:")
print(metrics_file)

print("\nConfusion matrix:")
print(confusion_matrix_file)

print("\nPredictions:")
print(prediction_file)

print("\n========== FINAL TABICLv2 RESULTS ==========\n")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")

print("\n========== TABICLv2 COMPLETE ==========\n")