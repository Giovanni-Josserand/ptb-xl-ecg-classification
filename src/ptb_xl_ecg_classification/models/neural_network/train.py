import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ptb_xl_ecg_classification.models.neural_network.load_split import load_split
from ptb_xl_ecg_classification.models.neural_network.network import ResNet1D
from ptb_xl_ecg_classification.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_PATH,
    DATA_PATH,
    DEVICE,
)


def train():
    nn_output_path = OUTPUT_PATH / "neural_network"
    nn_output_path.mkdir(parents=True, exist_ok=True)
    model_path = nn_output_path / "neural_network_model.pkl"

    print("Device:", DEVICE)
    print("Starting ResNet1D training...\n")

    train_loader, val_loader = create_data_loaders()
    model = ResNet1D().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": best_epoch,
                    "val_loss": best_val_loss,
                },
                model_path,
            )

    print("\nTraining completed!")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", round(best_val_loss, 4))
    print(f"Model saved to: {model_path}")


def create_data_loaders():
    X_train, y_train = load_split(DATA_PATH, "train")
    X_val, y_val = load_split(DATA_PATH, "val")

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


if __name__ == "__main__":
    train()