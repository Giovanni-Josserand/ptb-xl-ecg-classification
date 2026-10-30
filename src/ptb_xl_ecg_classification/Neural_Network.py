
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


DATA_DIR = Path("data/ptb-xl/ptbxl_clean")
MODEL_PATH = Path("models/resnet1d_best.pth")

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


def load_split(split):
    X = np.load(DATA_DIR / f"X_{split}.npy", mmap_mode="r")
    y = np.load(DATA_DIR / f"y_{split}.npy", mmap_mode="r")

    # NumPy store (N, 1000, 12).
    # Conv1d want (N, 12, 1000).
    X = np.transpose(X, (0, 2, 1)).copy()

    # Convert to PyTorch tensors.
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(np.array(y)).float()

    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=12,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 64),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock1D(128, 256, stride=2),
            ResidualBlock1D(256, 256),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


model = ResNet1D().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

print(model)

with torch.no_grad():
    test_input = torch.randn(2, 12, 1000, device=DEVICE)
    test_output = model(test_input)

print("Test output shape:", test_output.shape)
       # expected torch.Size([2, 5])


def run_epoch(loader, training):
    if training:
        model.train()
    else:
        model.eval()

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


MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state,
                "classes": CLASSES,
                "epoch": epoch + 1,
                "val_loss": val_loss,
            },
            MODEL_PATH,
        )

print("Best validation loss:", round(best_val_loss, 4))
print("Best model saved to:", MODEL_PATH)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Best epoch:", checkpoint["epoch"])

all_probabilities = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)

        logits = model(X_batch)
        probabilities = torch.sigmoid(logits)

        all_probabilities.append(probabilities.cpu().numpy())
        all_labels.append(y_batch.numpy())

probabilities = np.concatenate(all_probabilities, axis=0)
true_labels = np.concatenate(all_labels, axis=0)

predictions = (probabilities >= 0.5).astype(np.int32)
true_labels = true_labels.astype(np.int32)


from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
)

print("\nClassification report:")
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

print(f"Macro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")
print(f"Hamming loss: {hamming_loss(true_labels, predictions):.4f}")
