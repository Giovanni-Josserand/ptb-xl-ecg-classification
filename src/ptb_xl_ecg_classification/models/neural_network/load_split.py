import numpy as np
import torch


def load_split(DATA_PATH, split):
    X = np.load(DATA_PATH / f"X_{split}.npy", mmap_mode="r")
    y = np.load(DATA_PATH / f"y_{split}.npy", mmap_mode="r")

    # NumPy store (N, 1000, 12).
    # Conv1d want (N, 12, 1000).
    X = np.transpose(X, (0, 2, 1)).copy()

    # Convert to PyTorch tensors.
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(np.array(y)).float()

    return X, y