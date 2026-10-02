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

# =========================================================
# PROJECT
# Adaptive Health Assessment and Decision Support System
# for Induction Motors Using Vibration Signal Processing
# and Machine Learning
# =========================================================

BASE_DIR = "machine_learning"

TRAIN_FILE = os.path.join(BASE_DIR, "train_data.csv")
TEST_FILE = os.path.join(BASE_DIR, "test_data.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_FILE = os.path.join(MODEL_DIR, "tabicl_model.pkl")
METRICS_FILE = os.path.join(RESULTS_DIR, "tabicl_metrics.csv")
CONFUSION_FILE = os.path.join(RESULTS_DIR, "tabicl_confusion_matrix.csv")
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "tabicl_predictions.csv")

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

TARGET = "fault_type"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


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

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

X_train = train_data[FEATURES]
y_train = train_data[TARGET]

X_test = test_data[FEATURES]
y_test = test_data[TARGET]

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))
print("Input features   :", len(FEATURES))
print("Target           :", TARGET)
print("Fault classes    :", y_train.nunique())

print("\nInput Features:")
for feature in FEATURES:
    print("-", feature)


# =========================================================
# DATA CHECK
# =========================================================

print("\n==================== DATA CHECK ====================\n")

training_missing = X_train.isnull().sum().sum()
testing_missing = X_test.isnull().sum().sum()

print("Training missing values:", training_missing)
print("Testing missing values :", testing_missing)

if training_missing == 0 and testing_missing == 0:
    print("Status: No missing values found.")


# =========================================================
# MODEL
# =========================================================

print("\n==================== MODEL ====================\n")

print("Model : TabICLv2")
print("Task  : Multiclass Fault Classification")

model = TabICLClassifier(
    n_estimators=8,
    random_state=42,
    device=None,
    verbose=True
)

print("Model initialized.")


# =========================================================
# TRAINING
# =========================================================

print("\n==================== TRAINING ====================\n")

print("Training model...")
model.fit(X_train, y_train)

print("Training completed.")


# =========================================================
# PREDICTION
# =========================================================

print("\n==================== PREDICTION ====================\n")

y_pred = model.predict(X_test)

print("Prediction completed.")


# =========================================================
# PERFORMANCE
# =========================================================

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

print("\n==================== PERFORMANCE ====================\n")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n==================== CLASSIFICATION REPORT ====================\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

class_labels = [
    "Ball",
    "Inner Race",
    "Normal",
    "Outer Race"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_labels
)

print("\n==================== CONFUSION MATRIX ====================\n")

print("Classes:", class_labels)
print()
print(cm)


# =========================================================
# SAVE RESULTS
# =========================================================

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)

cm_df = pd.DataFrame(
    cm,
    index=class_labels,
    columns=class_labels
)

cm_df.to_csv(
    CONFUSION_FILE
)

predictions_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

model.save(MODEL_FILE)


# =========================================================
# SUMMARY
# =========================================================

correct = (y_test == y_pred).sum()
incorrect = len(y_test) - correct

print("\n==================== SUMMARY ====================\n")

print("Model             : TabICLv2")
print("Training samples  :", len(X_train))
print("Testing samples   :", len(X_test))
print("Features used     :", len(FEATURES))
print("Correct predictions   :", correct)
print("Incorrect predictions :", incorrect)

print("\n==================== PROCESS COMPLETE ====================\n")