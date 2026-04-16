import os
import json
import joblib
import torch
import torch.nn as nn
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "is_fraud_model.pth")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.json")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test_scaled.csv")


class FraudModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def load_feature_columns(path=FEATURE_COLUMNS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature columns file not found: {path}")
    return joblib.load(path)


def load_threshold(path=THRESHOLD_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Threshold file not found: {path}")

    with open(path, "r") as f:
        threshold_info = json.load(f)

    return float(threshold_info["threshold"])


def load_test_data(test_path=TEST_DATA_PATH):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data file not found: {test_path}")

    df = pd.read_csv(test_path)

    if "is_fraud" not in df.columns:
        raise ValueError("Target column 'is_fraud' not found in test dataset.")

    X_test = df.drop(columns=["is_fraud"]).astype("float32")
    y_test = df["is_fraud"].astype("int32")

    return X_test, y_test


def load_trained_model(input_dim, model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = FraudModel(input_dim=input_dim)

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # If you saved model.state_dict()
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        # If you accidentally saved the whole model object
        model = checkpoint

    model.eval()
    return model


def predict_one_case(model, X_test, y_test, threshold, row_index, case_name):
    sample_df = X_test.loc[[row_index]]
    actual_label = int(y_test.loc[row_index])

    sample_tensor = torch.tensor(sample_df.values, dtype=torch.float32)

    with torch.no_grad():
        fraud_probability = float(model(sample_tensor).item())

    predicted_label = int(fraud_probability >= threshold)

    print(f"\n=== {case_name} ===")
    print(f"Row index: {row_index}")
    print(f"Actual label: {actual_label}")
    print(f"Predicted probability: {fraud_probability:.4f}")
    print(f"Predicted label: {predicted_label}")

    if predicted_label == 1:
        print("Prediction: FRAUD")
    else:
        print("Prediction: NOT FRAUD")

    if predicted_label == actual_label:
        print("Result: CORRECT")
    else:
        print("Result: WRONG")


if __name__ == "__main__":
    print("Loading feature columns...")
    feature_columns = load_feature_columns()

    print("Loading threshold...")
    threshold = 0.3

    print("Loading test set...")
    X_test, y_test = load_test_data()

    # ensure correct feature order
    X_test = X_test[feature_columns]

    print("Loading trained PyTorch model...")
    model = load_trained_model(input_dim=X_test.shape[1])

    print(f"\nUsing dataset: {TEST_DATA_PATH}")
    print("Test features shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)

    normal_indices = y_test[y_test == 0].index
    fraud_indices = y_test[y_test == 1].index

    if len(normal_indices) == 0:
        raise ValueError("No normal cases found in test set.")
    if len(fraud_indices) == 0:
        raise ValueError("No fraud cases found in test set.")

    normal_index = normal_indices[0]
    fraud_index = fraud_indices[0]

    predict_one_case(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        row_index=normal_index,
        case_name="Normal Test Case"
    )

    predict_one_case(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        row_index=fraud_index,
        case_name="Fraud Test Case"
    )
