import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

from ptb_xl_ecg_classification.models.neural_network.load_split import load_split
from ptb_xl_ecg_classification.models.neural_network.network import ResNet1D
from ptb_xl_ecg_classification.config import (
    BATCH_SIZE,
    OUTPUT_PATH,
    DATA_PATH,
    DEVICE,
    CLASSES,
)


def load_data(split):
    X, y = load_split(DATA_PATH, split)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader


def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = ResNet1D().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(model, loader):
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            probabilities = torch.sigmoid(logits)

            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(y_batch.numpy())

    probabilities = np.concatenate(all_probabilities, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)
    predictions = (probabilities >= 0.5).astype(np.int32)
    true_labels = true_labels.astype(np.int32)

    return true_labels, predictions, probabilities


def get_metrics(y_true, y_pred, y_score):
    per_class = {}
    for i, cls in enumerate(CLASSES):
        p, r, f, s = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        try:
            roc = roc_auc_score(y_true[:, i], y_score[:, i])
        except ValueError:
            roc = 0.0
        try:
            pr = average_precision_score(y_true[:, i], y_score[:, i])
        except ValueError:
            pr = 0.0

        per_class[cls] = {
            "support": int(s) if s is not None else 0,
            "precision": float(p) if p is not None else 0.0,
            "recall": float(r) if r is not None else 0.0,
            "f1": float(f) if f is not None else 0.0,
            "roc_auc": float(roc),
            "pr_auc": float(pr)
        }

    _, macro_p, macro_r, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    _, _, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )

    macro_pr_auc = float(np.mean([per_class[c]["pr_auc"] for c in CLASSES]))

    return {
        "per_class": per_class,
        "macro_f1": float(macro_f1) if macro_f1 is not None else 0.0,
        "micro_f1": float(micro_f1) if micro_f1 is not None else 0.0,
        "macro_precision": float(macro_p) if macro_p is not None else 0.0,
        "macro_recall": float(macro_r) if macro_r is not None else 0.0,
        "macro_pr_auc": macro_pr_auc
    }


def main():
    nn_output_path = OUTPUT_PATH / "neural_network"
    nn_output_path.mkdir(parents=True, exist_ok=True)
    model_path = nn_output_path / "neural_network_model.pkl"
    metrics_path = nn_output_path / "metrics.json"

    model = load_model(model_path)

    val_loader = load_data("val")
    y_val, val_pred, val_score = predict(model, val_loader)

    test_loader = load_data("test")
    y_test, test_pred, test_score = predict(model, test_loader)

    feature_names = [f"channel_{i+1}_time" for i in range(12)]

    metrics = {
        "classes": CLASSES,
        "feature_count": 12,
        "feature_names": feature_names,
        "validation": get_metrics(y_val, val_pred, val_score),
        "test": get_metrics(y_test, test_pred, test_score)
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()