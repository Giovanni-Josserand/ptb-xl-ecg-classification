import numpy as np

DATASET_CLEAN_PATH = r"path\\to\\ptbxl_clean\\"

X_train = np.load(
    f"{DATASET_CLEAN_PATH}\X_train.npy",
    mmap_mode="r"
)
y_train = np.load(
    f"{DATASET_CLEAN_PATH}\y_train.npy",
    mmap_mode="r"
)

print("ECG data:", X_train.shape)
print("Labels:", y_train.shape)
print("Data type:", X_train.dtype)
print("All finite:", np.isfinite(X_train[0]).all())
print("First ECG labels:", y_train[0])

for split in ["train", "val", "test"]:
    X = np.load(f"{DATASET_CLEAN_PATH}/X_{split}.npy", mmap_mode="r")
    y = np.load(f"{DATASET_CLEAN_PATH}/y_{split}.npy", mmap_mode="r")

    print(split, X.shape, y.shape)

classes = ["NORM", "MI", "STTC", "CD", "HYP"]
y = np.load(f"{DATASET_CLEAN_PATH}/y_train.npy")

for i, name in enumerate(classes):
    print(f"{name}: {int(y[:, i].sum())} positive recordings")