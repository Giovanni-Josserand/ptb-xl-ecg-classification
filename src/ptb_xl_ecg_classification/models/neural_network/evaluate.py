from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
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


def load_test_data():

    X_test, y_test = load_split(DATA_PATH, "test")

    test_dataset = TensorDataset(
        X_test,
        y_test,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return test_loader


def load_model(model_path):

    model = ResNet1D().to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully.")
    print("Best epoch:", checkpoint["epoch"])
    print("Validation loss:", checkpoint["val_loss"])

    return model


def predict(model, test_loader):

    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)

            probabilities = torch.sigmoid(logits)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                y_batch.numpy()
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    true_labels = np.concatenate(
        all_labels,
        axis=0,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(np.int32)

    true_labels = true_labels.astype(np.int32)

    return true_labels, predictions, probabilities


def evaluate(true_labels, predictions):

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            true_labels,
            predictions,
            target_names=CLASSES,
            zero_division=0,
        )
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        true_labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    loss = hamming_loss(
        true_labels,
        predictions,
    )

    print("\n========== OVERALL METRICS ==========")

    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Micro F1:     {micro_f1:.4f}")
    print(f"Hamming loss: {loss:.4f}")

    print("\n========== CONFUSION MATRICES ==========")

    matrices = multilabel_confusion_matrix(
        true_labels,
        predictions,
    )

    for class_name, matrix in zip(CLASSES, matrices):

        print(f"\nClass: {class_name}")

        print(matrix)


def main():
    print("Starting ResNet1D evaluation...")
    print("Device:", DEVICE)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_PATH / "resnet1d_best.pth"

    test_loader = load_test_data()

    model = load_model(model_path)

    true_labels, predictions, probabilities = predict(
        model,
        test_loader,
    )

    evaluate(
        true_labels,
        predictions,
    )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
