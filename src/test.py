import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/is_fraud_model.pth"
TEST_PATH = "data/processed/test_scaled.csv"
OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "top_test_case_predictions.csv")

TARGET_COL = "is_fraud"
THRESHOLD = 0.30
TOP_K = 5


# =========================
# MODEL
# =========================
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


# =========================
# LOAD DATA
# =========================
def load_test_data():
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

    df = pd.read_csv(TEST_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {TEST_PATH}")

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    return df, X, y


# =========================
# LOAD MODEL
# =========================
def load_model(input_dim):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FraudNet(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device


# =========================
# PREDICT
# =========================
def predict(model, device, X_df):
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    preds = (probs >= THRESHOLD).astype(int)
    return probs, preds


# =========================
# REPORT HELPERS
# =========================
def assign_outcome(actual, pred):
    if actual == 1 and pred == 1:
        return "TRUE POSITIVE"
    elif actual == 0 and pred == 1:
        return "FALSE POSITIVE"
    elif actual == 0 and pred == 0:
        return "TRUE NEGATIVE"
    else:
        return "FALSE NEGATIVE"


def prediction_confidence(prob, pred):
    return prob if pred == 1 else (1 - prob)


def confidence_label(conf):
    if conf >= 0.90:
        return "Very High"
    elif conf >= 0.75:
        return "High"
    elif conf >= 0.60:
        return "Moderate"
    return "Low"


def print_divider():
    print("=" * 140)


def print_section(title, df):
    print_divider()
    print(title.center(140))
    print_divider()

    if df.empty:
        print("No cases in this section.\n")
        return

    cols_to_show = [
        "row_id",
        "actual_label_name",
        "predicted_label_name",
        "predicted_probability",
        "prediction_confidence",
        "confidence_label",
        "outcome"
    ]

    display_df = df[cols_to_show].copy()
    display_df["predicted_probability"] = display_df["predicted_probability"].map(lambda x: f"{x:.6f}")
    display_df["prediction_confidence"] = display_df["prediction_confidence"].map(lambda x: f"{x:.6f}")

    print(display_df.to_string(index=False))
    print()


def print_summary(results):
    tp = (results["outcome"] == "TRUE POSITIVE").sum()
    fp = (results["outcome"] == "FALSE POSITIVE").sum()
    tn = (results["outcome"] == "TRUE NEGATIVE").sum()
    fn = (results["outcome"] == "FALSE NEGATIVE").sum()

    total = len(results)
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    print_divider()
    print("TEST SET CASE REVIEW SUMMARY".center(140))
    print_divider()
    print(f"Threshold   : {THRESHOLD:.2f}")
    print(f"Total cases : {total}")
    print(f"TP          : {tp}")
    print(f"FP          : {fp}")
    print(f"TN          : {tn}")
    print(f"FN          : {fn}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print()


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    full_df, X_test, y_test = load_test_data()

    model, device = load_model(X_test.shape[1])
    probs, preds = predict(model, device, X_test)

    results = full_df.copy().reset_index(drop=True)
    results["row_id"] = results.index
    results["actual_label"] = y_test.values
    results["predicted_label"] = preds
    results["predicted_probability"] = probs
    results["prediction_confidence"] = [
        prediction_confidence(p, pred) for p, pred in zip(probs, preds)
    ]
    results["confidence_label"] = results["prediction_confidence"].apply(confidence_label)
    results["actual_label_name"] = results["actual_label"].map({0: "Not Fraud", 1: "Fraud"})
    results["predicted_label_name"] = results["predicted_label"].map({0: "Not Fraud", 1: "Fraud"})
    results["outcome"] = [
        assign_outcome(a, p) for a, p in zip(results["actual_label"], results["predicted_label"])
    ]

    print_summary(results)

    tp_top5 = results[results["outcome"] == "TRUE POSITIVE"] \
        .sort_values("predicted_probability", ascending=False).head(TOP_K)

    fp_top5 = results[results["outcome"] == "FALSE POSITIVE"] \
        .sort_values("predicted_probability", ascending=False).head(TOP_K)

    fn_top5 = results[results["outcome"] == "FALSE NEGATIVE"] \
        .sort_values("predicted_probability", ascending=True).head(TOP_K)

    tn_top5 = results[results["outcome"] == "TRUE NEGATIVE"] \
        .sort_values("prediction_confidence", ascending=False).head(TOP_K)

    print_section("TOP 5 TRUE POSITIVES (Correct Fraud Catches)", tp_top5)
    print_section("TOP 5 FALSE POSITIVES (Wrongly Flagged as Fraud)", fp_top5)
    print_section("TOP 5 FALSE NEGATIVES (Missed Fraud Cases)", fn_top5)
    print_section("TOP 5 TRUE NEGATIVES (Correct Genuine Cases)", tn_top5)


    results.to_csv(OUTPUT_PATH, index=False)
    print_divider()
    print(f"Saved full results to: {OUTPUT_PATH}")
    print_divider()


if __name__ == "__main__":
    main()
