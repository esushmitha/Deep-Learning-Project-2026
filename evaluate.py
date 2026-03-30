import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


def load_test_data(test_path="data/processed/test_scaled.csv"):
    """
    Load scaled test data and separate features/target.
    """
    df = pd.read_csv(test_path)

    if "is_fraud" not in df.columns:
        raise ValueError("Target column 'is_fraud' not found in test dataset.")

    X_test = df.drop(columns=["is_fraud"]).astype(np.float32)
    y_test = df["is_fraud"].astype(np.int32)

    return X_test, y_test


class FraudNet(nn.Module):
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

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


def load_trained_model(model_path="models/is_fraud_model.pth", input_dim=None):
    """
    Load trained PyTorch model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if input_dim is None:
        raise ValueError("input_dim must be provided to load the PyTorch model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FraudNet(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device


def load_threshold(threshold_path="models/threshold.json"):
    """
    Load threshold selected on validation set during training.
    """
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

    with open(threshold_path, "r") as f:
        threshold_info = json.load(f)

    if "threshold" not in threshold_info:
        raise ValueError("Key 'threshold' not found in threshold.json")

    threshold = float(threshold_info["threshold"])
    return threshold, threshold_info


def predict_probabilities(model, X_test, device, batch_size=4096):
    """
    Predict probabilities from PyTorch model.
    """
    X_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

    probs = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size]
            logits = model(batch)
            batch_probs = torch.sigmoid(logits).cpu().numpy().ravel()
            probs.extend(batch_probs)

    return np.array(probs)


def evaluate_model(model, X_test, y_test, threshold, device):
    """
    Evaluate trained model on the test set using a pre-chosen threshold.
    """
    # Predict probabilities
    y_prob = predict_probabilities(model, X_test, device=device)

    # Apply saved threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    print("\n=== Final Test Set Evaluation ===")
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\n=== Confusion Matrix ===")
    print(cm)

    print("\n=== Classification Report ===")
    print(report)

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_prob": y_prob,
        "y_pred": y_pred
    }


def save_predictions(X_test, y_test, y_prob, y_pred, output_path="data/processed/test_predictions.csv"):
    """
    Save test predictions for later analysis.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    predictions_df = X_test.copy()
    predictions_df["is_fraud"] = y_test.values
    predictions_df["predicted_probability"] = y_prob
    predictions_df["predicted_label"] = y_pred

    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")


def save_metrics(results, output_path="models/test_metrics.json"):
    """
    Save final test metrics to JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics_to_save = {
        "threshold": results["threshold"],
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1_score": results["f1_score"],
        "roc_auc": results["roc_auc"],
        "pr_auc": results["pr_auc"],
        "confusion_matrix": results["confusion_matrix"],
        "classification_report": results["classification_report"]
    }

    with open(output_path, "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print(f"Test metrics saved to {output_path}")


if __name__ == "__main__":
    test_data_path = "data/processed/test_scaled.csv"
    model_path = "models/is_fraud_model.pth"
    threshold_path = "models/threshold.json"
    prediction_output_path = "data/processed/test_predictions.csv"
    metrics_output_path = "models/test_metrics.json"

    # Load test data
    X_test, y_test = load_test_data(test_data_path)

    print("Test features shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)

    # Load trained model
    model, device = load_trained_model(
        model_path=model_path,
        input_dim=X_test.shape[1]
    )

    # Load threshold chosen from validation set
    threshold = 0.3

    print("\n=== Using Manual Threshold ===")
    print(f"Threshold: {threshold}")

    # Final evaluation on test set
    results = evaluate_model(model, X_test, y_test, threshold=threshold, device=device)

    # Save predictions
    save_predictions(
        X_test=X_test,
        y_test=y_test,
        y_prob=results["y_prob"],
        y_pred=results["y_pred"],
        output_path=prediction_output_path
    )

    # Save metrics
    save_metrics(results, output_path=metrics_output_path)

    print("\nEvaluation complete.")