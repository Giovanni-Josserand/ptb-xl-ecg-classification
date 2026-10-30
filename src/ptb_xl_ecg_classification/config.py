from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


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


# RANDOM FOREST CONFIGURATION
RANDOM_FOREST_OUTPUT_PATH = OUTPUT_PATH / "random_forest"
RANDOM_FOREST_N_ESTIMATORS = 300
RANDOM_FOREST_RANDOM_STATE = 42
RANDOM_FOREST_CHUNK_SIZE = 512


# COMPUTING DEVICE
DEVICE = (
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch is not None
    else "cpu"
)

