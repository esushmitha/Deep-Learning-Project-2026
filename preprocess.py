import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath="data/raw/fraudTest.csv"):
    """
    Load raw fraud dataset.
    """
    df = pd.read_csv(filepath)
    print("Raw data loaded.")
    print("Shape:", df.shape)
    print(df.head())
    return df


def clean_and_engineer_features(df):
    """
    Basic cleaning and feature engineering.
    """
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Drop rows with missing target
    df = df.dropna(subset=["is_fraud"]).copy()

    # Convert date columns
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # Feature engineering
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day"] = df["trans_date_trans_time"].dt.day
    df["trans_month"] = df["trans_date_trans_time"].dt.month
    df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_weekend"] = df["trans_dayofweek"].isin([5, 6]).astype(int)

    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25

    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 +
        (df["long"] - df["merch_long"]) ** 2
    )

    df["amt_log"] = np.log1p(df["amt"])

    # Drop leakage / irrelevant / ID-like columns
    drop_cols = [
        "trans_date_trans_time",
        "dob",
        "trans_num",
        "cc_num",
        "first",
        "last",
        "street"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def fill_missing_values(df):
    """
    Fill missing values before splitting features/target.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "is_fraud" in numeric_cols:
        numeric_cols.remove("is_fraud")

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    df = df.dropna().copy()
    return df


def split_data(X, y, random_state=42):
    """
    Split into:
    - 80% train
    - 10% validation
    - 10% test
    """
    # First split: 90% temp, 10% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=random_state,
        stratify=y
    )

    # Second split: 1/9 of remaining 90% goes to validation
    # This gives 10% validation overall and 80% training overall
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=1/9,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_label_encoders(X_train, categorical_cols):
    """
    Fit label encoders on training data only.
    """
    X_train = X_train.copy()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        label_encoders[col] = le

    return X_train, label_encoders


def transform_with_label_encoders(X, categorical_cols, label_encoders):
    """
    Transform validation/test data using encoders fitted on train only.
    Unseen categories are mapped to 'Unknown'.
    """
    X = X.copy()

    for col in categorical_cols:
        le = label_encoders[col]
        values = X[col].astype(str)

        # Add "Unknown" class if not already present
        if "Unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "Unknown")

        known_classes = set(le.classes_)
        values = values.apply(lambda x: x if x in known_classes else "Unknown")
        X[col] = le.transform(values)

    return X


def scale_features(X_train, X_val, X_test):
    """
    Fit scaler on training data only, then transform validation and test.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_dataframes(X_df, y_series, filepath):
    """
    Save features + target as one CSV.
    """
    df_out = X_df.copy()
    df_out["is_fraud"] = y_series.values
    df_out.to_csv(filepath, index=False)


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Load raw data
    df = load_data("data/raw/fraudTest.csv")

    # 2. Clean + feature engineering
    df = clean_and_engineer_features(df)

    # 3. Fill missing values
    df = fill_missing_values(df)

    # 4. Separate features and target
    X = df.drop(columns=["is_fraud"]).copy()
    y = df["is_fraud"].astype(np.int32).copy()

    print("\nTarget distribution:")
    print(y.value_counts())
    print(y.value_counts(normalize=True))

    # Save fully cleaned unencoded dataset for reference
    clean_df = X.copy()
    clean_df["is_fraud"] = y.values
    clean_df.to_csv("data/processed/clean_dataset.csv", index=False)

    # 5. Split data into train / val / test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("\nSplit shapes:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    print("\nClass proportions:")
    print("Train:")
    print(y_train.value_counts(normalize=True))
    print("Validation:")
    print(y_val.value_counts(normalize=True))
    print("Test:")
    print(y_test.value_counts(normalize=True))

    # 6. Encode categoricals using train only
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    print("\nCategorical columns encoded:")
    print(categorical_cols)

    X_train, label_encoders = fit_label_encoders(X_train, categorical_cols)
    X_val = transform_with_label_encoders(X_val, categorical_cols, label_encoders)
    X_test = transform_with_label_encoders(X_test, categorical_cols, label_encoders)

    # 7. Save unscaled encoded splits
    save_dataframes(X_train, y_train, "data/processed/train_unscaled.csv")
    save_dataframes(X_val, y_val, "data/processed/val_unscaled.csv")
    save_dataframes(X_test, y_test, "data/processed/test_unscaled.csv")

    # 8. Scale using train only
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # 9. Save scaled splits
    save_dataframes(X_train_scaled_df, y_train, "data/processed/train_scaled.csv")
    save_dataframes(X_val_scaled_df, y_val, "data/processed/val_scaled.csv")
    save_dataframes(X_test_scaled_df, y_test, "data/processed/test_scaled.csv")

    # 10. Save preprocessing objects
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(list(X_train.columns), "models/feature_columns.pkl")

    print("\nPreprocessing complete.")
    print("Saved:")
    print("- data/processed/clean_dataset.csv")
    print("- data/processed/train_unscaled.csv")
    print("- data/processed/val_unscaled.csv")
    print("- data/processed/test_unscaled.csv")
    print("- data/processed/train_scaled.csv")
    print("- data/processed/val_scaled.csv")
    print("- data/processed/test_scaled.csv")
    print("- models/scaler.pkl")
    print("- models/label_encoders.pkl")
    print("- models/feature_columns.pkl")