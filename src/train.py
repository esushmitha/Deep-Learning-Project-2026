import os
import json
import copy
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def load_data_splits(
    train_path="data/processed/train_scaled.csv",
    val_path="data/processed/val_scaled.csv"
):
    """
    Load pre-split train and validation datasets.
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    if "is_fraud" not in train_df.columns:
        raise ValueError("Target column 'is_fraud' not found in training dataset.")
    if "is_fraud" not in val_df.columns:
        raise ValueError("Target column 'is_fraud' not found in validation dataset.")

    X_train = train_df.drop(columns=["is_fraud"]).astype(np.float32)
    y_train = train_df["is_fraud"].astype(np.float32)

    X_val = val_df.drop(columns=["is_fraud"]).astype(np.float32)
    y_val = val_df["is_fraud"].astype(np.float32)

    return X_train, y_train, X_val, y_val


def compute_class_weights(y_train, max_fraud_weight=10.0):
    """
    Compute class weights to handle class imbalance.
    Fraud weight is capped to avoid excessive false positives.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight = dict(zip(classes, weights))

    if 1 in class_weight:
        class_weight[1] = min(class_weight[1], max_fraud_weight)

    return class_weight


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification using logits.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        targets = targets.view(-1, 1)

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma

        loss = alpha_factor * modulating_factor * bce_loss
        return loss.mean()


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

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)


def build_model(input_dim):
    """
    Build feedforward neural network for fraud detection.
    """
    return FraudNet(input_dim)


def find_best_threshold(y_true, y_prob):
    """
    Find best threshold based on F1-score using validation data.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    precisions = precisions[:-1]
    recalls = recalls[:-1]

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)

    best_threshold = float(thresholds[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return best_threshold, best_precision, best_recall, best_f1


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def predict_probabilities(model, loader, device):
    model.eval()
    probs_list = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            probs_list.extend(probs)

    return np.array(probs_list)


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    model_save_path="models/is_fraud_model.pth",
    epochs=50,
    batch_size=1024,
    learning_rate=3e-4,
    weight_decay=1e-4,
    patience=8
):
    """
    Train the model using train and validation sets only.
    """
    os.makedirs("models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    print("\nClass distribution:")
    print("Train:")
    print(y_train.value_counts(normalize=True))
    print("Validation:")
    print(y_val.value_counts(normalize=True))

    class_weight = compute_class_weights(y_train)
    print("\nClass weights:", class_weight)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(input_dim=X_train.shape[1]).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=0.25)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size_current = X_batch.size(0)
            running_loss += loss.item() * batch_size_current
            total_samples += batch_size_current

        train_loss = running_loss / total_samples
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_save_path)
            print(f"Best model saved to {model_save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Tune threshold on validation set only
    y_val_prob = predict_probabilities(model, val_loader, device)
    best_threshold, best_precision, best_recall, best_f1 = find_best_threshold(
        y_val.values,
        y_val_prob
    )

    threshold_info = {
        "threshold": best_threshold,
        "precision": best_precision,
        "recall": best_recall,
        "f1_score": best_f1
    }

    with open("models/threshold.json", "w") as f:
        json.dump(threshold_info, f, indent=4)

    return {
        "model": model,
        "history": history,
        "class_weight": class_weight,
        "threshold_info": threshold_info
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Reproducibility
    set_seed(42)

    # Load pre-split train and validation data
    X_train, y_train, X_val, y_val = load_data_splits(
        train_path="data/processed/train_scaled.csv",
        val_path="data/processed/val_scaled.csv"
    )

    print("Training features shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Validation features shape:", X_val.shape)
    print("Validation labels shape:", y_val.shape)

    # Train model
    results = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_save_path="models/is_fraud_model.pth",
        epochs=50,
        batch_size=1024
    )

    print("\nTraining complete.")
    print("Model saved to models/is_fraud_model.pth")
    print("Threshold saved to models/threshold.json")