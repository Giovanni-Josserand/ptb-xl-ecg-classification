from pathlib import Path

import torch


# PROJECT PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "ptb-xl" / "ptbxl_clean"
OUTPUT_PATH = PROJECT_ROOT / "data" / "output"


# DATASET CONFIGURATION
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# TRAINING CONFIGURATION
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3


# COMPUTING DEVICE
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

