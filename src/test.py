import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================
# CONFIG
# =========================
MODEL_PATH = "models/is_fraud_model.pth"
SCALER_PATH = "models/scaler.pkl"
ENCODERS_PATH = "models/label_encoders.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"

OUTPUT_DIR = "output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "manual_test_case_predictions.csv")

THRESHOLD = 0.30


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
# MANUAL TEST CASES
# =========================
def build_manual_test_cases():
    """
    Create realistic manual test cases in raw format.
    Keep columns similar to your original fraud dataset.
    """
    rows = [
        {
            "trans_date_trans_time": "2020-06-21 09:15:00",
            "cc_num": 4000001234567890,
            "merchant": "fraud_Starbucks",
            "category": "food_dining",
            "amt": 6.80,
            "first": "Alicia",
            "last": "Tan",
            "gender": "F",
            "street": "12 Jurong West St 41",
            "city": "Singapore",
            "state": "SG",
            "zip": 640012,
            "lat": 1.3496,
            "long": 103.7069,
            "city_pop": 5637000,
            "job": "Teacher",
            "dob": "1996-04-18",
            "trans_num": "case_001",
            "unix_time": 1592730900,
            "merch_lat": 1.3504,
            "merch_long": 103.7075,
            "case_note": "Small local coffee purchase"
        },
        {
            "trans_date_trans_time": "2020-06-21 13:40:00",
            "cc_num": 4000001234567891,
            "merchant": "fraud_NTUC_FairPrice",
            "category": "grocery_pos",
            "amt": 24.90,
            "first": "Marcus",
            "last": "Lim",
            "gender": "M",
            "street": "88 Bedok North Ave 4",
            "city": "Singapore",
            "state": "SG",
            "zip": 460088,
            "lat": 1.3329,
            "long": 103.9142,
            "city_pop": 5637000,
            "job": "Engineer",
            "dob": "1989-11-02",
            "trans_num": "case_002",
            "unix_time": 1592746800,
            "merch_lat": 1.3337,
            "merch_long": 103.9150,
            "case_note": "Typical grocery purchase"
        },
        {
            "trans_date_trans_time": "2020-06-23 02:10:00",
            "cc_num": 4000001234567892,
            "merchant": "fraud_Amazon",
            "category": "shopping_net",
            "amt": 329.99,
            "first": "Ryan",
            "last": "Ong",
            "gender": "M",
            "street": "9 Woodlands Dr 50",
            "city": "Singapore",
            "state": "SG",
            "zip": 730009,
            "lat": 1.4382,
            "long": 103.7932,
            "city_pop": 5637000,
            "job": "Software Developer",
            "dob": "1998-12-09",
            "trans_num": "case_003",
            "unix_time": 1592878200,
            "merch_lat": 34.0522,
            "merch_long": -118.2437,
            "case_note": "Night-time online purchase with far merchant"
        },
        {
            "trans_date_trans_time": "2020-06-23 04:45:00",
            "cc_num": 4000001234567893,
            "merchant": "fraud_Apple",
            "category": "shopping_net",
            "amt": 899.00,
            "first": "Jia",
            "last": "Chen",
            "gender": "F",
            "street": "101 Punggol Field",
            "city": "Singapore",
            "state": "SG",
            "zip": 820101,
            "lat": 1.3995,
            "long": 103.9072,
            "city_pop": 5637000,
            "job": "Student",
            "dob": "2000-05-11",
            "trans_num": "case_004",
            "unix_time": 1592887500,
            "merch_lat": 37.3349,
            "merch_long": -122.0090,
            "case_note": "High-value electronics order at unusual hour"
        },
        {
            "trans_date_trans_time": "2020-06-23 03:05:00",
            "cc_num": 4000001234567894,
            "merchant": "fraud_Bitcoin_Exchange",
            "category": "misc_net",
            "amt": 2450.00,
            "first": "Aaron",
            "last": "Lee",
            "gender": "M",
            "street": "7 Hougang Ave 8",
            "city": "Singapore",
            "state": "SG",
            "zip": 530007,
            "lat": 1.3721,
            "long": 103.8930,
            "city_pop": 5637000,
            "job": "Sales Executive",
            "dob": "1985-06-27",
            "trans_num": "case_005",
            "unix_time": 1592881500,
            "merch_lat": 40.7128,
            "merch_long": -74.0060,
            "case_note": "Very high-value online transaction"
        },
        {
            "trans_date_trans_time": "2020-06-23 03:12:00",
            "cc_num": 4000001234567895,
            "merchant": "fraud_Luxury_Watches",
            "category": "shopping_net",
            "amt": 3875.49,
            "first": "Priya",
            "last": "Nair",
            "gender": "F",
            "street": "55 Serangoon North Ave 4",
            "city": "Singapore",
            "state": "SG",
            "zip": 550055,
            "lat": 1.3700,
            "long": 103.8720,
            "city_pop": 5637000,
            "job": "Consultant",
            "dob": "1991-03-21",
            "trans_num": "case_006",
            "unix_time": 1592881920,
            "merch_lat": 51.5074,
            "merch_long": -0.1278,
            "case_note": "Luxury online purchase in middle of night"
        }
    ]
    return pd.DataFrame(rows)


# =========================
# PREPROCESS
# =========================
def feature_engineering(df):
    df = df.copy()

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    today = pd.Timestamp("2020-06-23")
    df["age"] = ((today - df["dob"]).dt.days / 365.25).astype(int)

    df["lat_diff"] = (df["lat"] - df["merch_lat"]).abs()
    df["long_diff"] = (df["long"] - df["merch_long"]).abs()
    df["distance"] = np.sqrt(df["lat_diff"] ** 2 + df["long_diff"] ** 2)

    return df


def apply_label_encoders(df, label_encoders):
    df = df.copy()

    for col, encoder in label_encoders.items():
        if col not in df.columns:
            continue

        values = df[col].astype(str)

        known_classes = set(encoder.classes_)
        if "Unknown" in known_classes:
            values = values.apply(lambda x: x if x in known_classes else "Unknown")
        else:
            # If encoder has no Unknown class, map unseen values to first class to avoid crash
            fallback = encoder.classes_[0]
            values = values.apply(lambda x: x if x in known_classes else fallback)

        df[col] = encoder.transform(values)

    return df


def preprocess_manual_cases(raw_df):
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    df = feature_engineering(raw_df)

    # Drop columns not used by model after feature engineering
    drop_cols = ["trans_date_trans_time", "dob", "first", "last", "street", "trans_num", "case_note"]
    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop)

    df = apply_label_encoders(df, label_encoders)

    # Fill missing columns if needed
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep correct order only
    df = df[feature_columns]

    scaled_array = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns)

    return scaled_df


# =========================
# PREDICT
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


def predict(model, device, X_df):
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    preds = (probs >= THRESHOLD).astype(int)
    return probs, preds


# =========================
# REPORTING
# =========================
def risk_level(prob):
    if prob >= 0.80:
        return "HIGH"
    if prob >= 0.50:
        return "MEDIUM"
    return "LOW"


def confidence_label(prob):
    if prob >= 0.90 or prob <= 0.10:
        return "Very High"
    if prob >= 0.75 or prob <= 0.25:
        return "High"
    if prob >= 0.60 or prob <= 0.40:
        return "Moderate"
    return "Low"


def explain_case(row):
    reasons = []

    if row["amt"] >= 1000:
        reasons.append("very high amount")
    elif row["amt"] >= 300:
        reasons.append("moderately high amount")

    if row["category"] in ["shopping_net", "misc_net"]:
        reasons.append("online transaction")

    hour = pd.to_datetime(row["trans_date_trans_time"]).hour
    if 0 <= hour <= 5:
        reasons.append("unusual transaction time")

    lat_diff = abs(row["lat"] - row["merch_lat"])
    long_diff = abs(row["long"] - row["merch_long"])
    if lat_diff > 1 or long_diff > 1:
        reasons.append("merchant is far from customer location")

    if not reasons:
        return "Looks like a normal transaction pattern."
    return "Flagged because of " + ", ".join(reasons) + "."


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_cases = build_manual_test_cases()
    X_manual = preprocess_manual_cases(raw_cases)

    model, device = load_model(X_manual.shape[1])
    probs, preds = predict(model, device, X_manual)

    results = raw_cases.copy()
    results["predicted_probability"] = probs
    results["predicted_label"] = preds
    results["predicted_label_name"] = results["predicted_label"].map({0: "Not Fraud", 1: "Fraud"})
    results["risk_level"] = results["predicted_probability"].apply(risk_level)
    results["confidence"] = results["predicted_probability"].apply(confidence_label)
    results["explanation"] = results.apply(explain_case, axis=1)

    results = results.sort_values("predicted_probability", ascending=False)

    display_cols = [
        "trans_num",
        "merchant",
        "category",
        "amt",
        "trans_date_trans_time",
        "case_note",
        "predicted_probability",
        "predicted_label_name",
        "risk_level",
        "confidence",
        "explanation",
    ]

    print("\n=== MANUAL TEST CASE RESULTS ===")
    print(results[display_cols].to_string(index=False))

    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()