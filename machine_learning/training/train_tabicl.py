import pandas as pd
from tabicl import TabICLClassifier

# Load the training and testing datasets
train = pd.read_csv("dataset/ml/train.csv")
test = pd.read_csv("dataset/ml/test.csv")

print("Training data shape:", train.shape)
print("Testing data shape:", test.shape)
print("Testing data shape:", test.shape)
# Select vibration features for machine learning
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

# Input features (X)
X_train = train[features]
X_test = test[features]

# Target labels (y)
y_train = train["fault_type"]
y_test = test["fault_type"]

print("Number of features:", len(features))
print("Feature names:", features)
print("Classes:", sorted(y_train.unique()))
# Create the TabICLv2 model
model = TabICLClassifier()

print("Training TabICLv2...")

# Train the model
model.fit(X_train, y_train)

print("TabICLv2 training completed.")