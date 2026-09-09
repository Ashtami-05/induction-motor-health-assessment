import os
import pandas as pd
import joblib

from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. FILE PATHS
# ==========================================

train_file = "machine_learning/train_data.csv"
test_file = "machine_learning/test_data.csv"

# Create folders for results and saved models
os.makedirs("machine_learning/models", exist_ok=True)
os.makedirs("machine_learning/results", exist_ok=True)


# ==========================================
# 2. LOAD TRAINING AND TESTING DATA
# ==========================================

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("\n========== DATA LOADED ==========\n")

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


# ==========================================
# 3. DEFINE ML FEATURES
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
# 4. CREATE X AND y
# ==========================================

X_train = train_df[features]
X_test = test_df[features]

y_train = train_df[target]
y_test = test_df[target]


print("\n========== ML DATA ==========\n")

print("Number of features:", len(features))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 5. ENCODE FAULT CLASSES
# ==========================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("\n========== FAULT CLASS ENCODING ==========\n")

for number, class_name in enumerate(label_encoder.classes_):
    print(number, "=", class_name)


# ==========================================
# 6. CREATE LIGHTGBM MODEL
# ==========================================

model = LGBMClassifier(
    objective="multiclass",
    num_class=4,
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\n========== MODEL TRAINING ==========\n")

print("Training LightGBM...")

model.fit(
    X_train,
    y_train_encoded
)

print("LightGBM training completed.")


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

print("\n========== PREDICTION ==========\n")

y_pred_encoded = model.predict(X_test)

print("Predictions completed.")


# ==========================================
# 9. CALCULATE PERFORMANCE METRICS
# ==========================================

accuracy = accuracy_score(
    y_test_encoded,
    y_pred_encoded
)

precision = precision_score(
    y_test_encoded,
    y_pred_encoded,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test_encoded,
    y_pred_encoded,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test_encoded,
    y_pred_encoded,
    average="weighted",
    zero_division=0
)


# ==========================================
# 10. DISPLAY PERFORMANCE
# ==========================================

print("\n========== LIGHTGBM PERFORMANCE ==========\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========\n")

report = classification_report(
    y_test_encoded,
    y_pred_encoded,
    target_names=label_encoder.classes_,
    zero_division=0
)

print(report)


# ==========================================
# 12. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test_encoded,
    y_pred_encoded
)

print("\n========== CONFUSION MATRIX ==========\n")

print(cm)


# ==========================================
# 13. SAVE CONFUSION MATRIX
# ==========================================

confusion_matrix_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

confusion_matrix_file = (
    "machine_learning/results/lightgbm_confusion_matrix.csv"
)

confusion_matrix_df.to_csv(confusion_matrix_file)


# ==========================================
# 14. FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========\n")

print(feature_importance)


# ==========================================
# 15. SAVE FEATURE IMPORTANCE
# ==========================================

feature_importance_file = (
    "machine_learning/results/lightgbm_feature_importance.csv"
)

feature_importance.to_csv(
    feature_importance_file,
    index=False
)


# ==========================================
# 16. SAVE MODEL
# ==========================================

model_file = "machine_learning/models/lightgbm_model.pkl"

joblib.dump(model, model_file)


# Save label encoder
encoder_file = "machine_learning/models/label_encoder.pkl"

joblib.dump(
    label_encoder,
    encoder_file
)


# ==========================================
# 17. SAVE PERFORMANCE METRICS
# ==========================================

metrics = pd.DataFrame({
    "Model": ["LightGBM"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1]
})

metrics_file = "machine_learning/results/lightgbm_metrics.csv"

metrics.to_csv(
    metrics_file,
    index=False
)


# ==========================================
# 18. FINAL OUTPUT
# ==========================================

print("\n========== FILES SAVED ==========\n")

print("Model:")
print(model_file)

print("\nLabel encoder:")
print(encoder_file)

print("\nMetrics:")
print(metrics_file)

print("\nConfusion matrix:")
print(confusion_matrix_file)

print("\nFeature importance:")
print(feature_importance_file)

print("\n========== LIGHTGBM TRAINING COMPLETE ==========\n")