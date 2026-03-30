import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from evaluate import load_test_data, load_trained_model, predict_probabilities


# match your current evaluate.py
TEST_DATA_PATH = "data/processed/test_scaled.csv"
MODEL_PATH = "models/is_fraud_model.pth"
THRESHOLD = 0.3

OUTPUT_DIR = "output/evaluation_plots"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_threshold_table(y_true, y_prob, max_points=200):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if len(thresholds) > max_points:
        idx = np.linspace(0, len(thresholds) - 1, max_points, dtype=int)
        thresholds = thresholds[idx]
        precisions = precisions[idx]
        recalls = recalls[idx]

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    rows = []
    for thr, prec, rec, f1_val in zip(thresholds, precisions, recalls, f1_scores):
        y_pred_thr = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thr).ravel()
        acc = accuracy_score(y_true, y_pred_thr)

        rows.append({
            "threshold": float(thr),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1_val),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        })

    return rows


def main():
    ensure_dir(OUTPUT_DIR)

    X_test, y_test = load_test_data(TEST_DATA_PATH)
    model, device = load_trained_model(MODEL_PATH, input_dim=X_test.shape[1])

    y_prob = predict_probabilities(model, X_test, device=device)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # save core metrics too
    metrics = {
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    with open(os.path.join(OUTPUT_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    threshold_rows = build_threshold_table(y_test, y_prob, max_points=200)

    thresholds = np.array([r["threshold"] for r in threshold_rows])
    accs = np.array([r["accuracy"] for r in threshold_rows])
    precs = np.array([r["precision"] for r in threshold_rows])
    recs = np.array([r["recall"] for r in threshold_rows])
    f1s = np.array([r["f1_score"] for r in threshold_rows])
    fps = np.array([r["false_positives"] for r in threshold_rows])
    tps = np.array([r["true_positives"] for r in threshold_rows])
    fns = np.array([r["false_negatives"] for r in threshold_rows])

    # 1. confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    disp.plot()
    plt.title(f"Confusion Matrix (threshold={THRESHOLD:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Precision-Recall curve
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recalls, precisions, label=f"PR AUC = {metrics['pr_auc']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_precision_recall_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. metrics vs threshold
    plt.figure()
    plt.plot(thresholds, accs, label="Accuracy")
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, f1s, label="F1-score")
    plt.axvline(THRESHOLD, linestyle="--", label=f"Chosen threshold = {THRESHOLD:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Metrics vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_metrics_vs_threshold.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 5. false positives vs threshold
    plt.figure()
    plt.plot(thresholds, fps)
    plt.axvline(THRESHOLD, linestyle="--", label=f"Chosen threshold = {THRESHOLD:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("False Positives")
    plt.title("False Positives vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_false_positives_vs_threshold.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 6. true positives / false negatives vs threshold
    plt.figure()
    plt.plot(thresholds, tps, label="True Positives")
    plt.plot(thresholds, fns, label="False Negatives")
    plt.axvline(THRESHOLD, linestyle="--", label=f"Chosen threshold = {THRESHOLD:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Count")
    plt.title("True Positives and False Negatives vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_tp_fn_vs_threshold.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 7. predicted probability by class
    plt.figure()
    plt.hist(y_prob[y_test == 0], bins=50, alpha=0.7, label="Actual non-fraud")
    plt.hist(y_prob[y_test == 1], bins=50, alpha=0.7, label="Actual fraud")
    plt.axvline(THRESHOLD, linestyle="--", label=f"Threshold = {THRESHOLD:.2f}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Predicted Probability Distribution by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "7_probability_distribution_by_class.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 8. metric bar chart
    labels = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    values = [metrics[k] for k in labels]
    plt.figure()
    plt.bar(labels, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "8_metric_bar_chart.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved 8 graphs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()