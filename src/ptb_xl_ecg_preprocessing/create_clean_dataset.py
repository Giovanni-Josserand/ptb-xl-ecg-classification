
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import wfdb

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from tqdm import tqdm


# ---------------- CONFIGURATION ----------------

DATASET_PATH = "path\\to\\datasets\\"

DATA_DIR = Path(f"{DATASET_PATH}\ptbxl")
OUTPUT_DIR = Path(f"{DATASET_PATH}\ptbxl_clean")

SAMPLING_RATE = 100
LOWPASS_CUTOFF = 40.0
HIGHPASS_CUTOFF = 0.5

# At 100 Hz, 50 Hz is the Nyquist frequency.
# Leave notch disabled for this configuration.
NOTCH_FREQUENCY = None

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ---------------- STEP 1: LOAD METADATA ----------------

def load_metadata(data_dir):
    metadata = pd.read_csv(
        data_dir / "ptbxl_database.csv",
        index_col="ecg_id",
    )
    statements = pd.read_csv(
        data_dir / "scp_statements.csv",
        index_col=0,
    )

    metadata["scp_codes"] = metadata["scp_codes"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
    )

    return metadata, statements


# ---------------- STEP 2: LOAD ECG ----------------

def load_ecg(row, data_dir, sampling_rate=100):
    column = "filename_lr" if sampling_rate == 100 else "filename_hr"

    if sampling_rate not in (100, 500):
        raise ValueError("Sampling rate must be 100 or 500 Hz.")

    record_path = data_dir / row[column]
    signal, fields = wfdb.rdsamp(str(record_path))

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError(f"Unexpected ECG shape: {signal.shape}")

    if not np.isfinite(signal).all():
        raise ValueError("ECG contains NaN or infinite values.")

    if fields["fs"] != sampling_rate:
        raise ValueError(
            f"Expected {sampling_rate} Hz, got {fields['fs']} Hz."
        )

    return signal, fields["fs"]


# ---------------- STEP 3: FILTER ECG ----------------

def remove_baseline_wander(signal, fs):
    sos = butter(
        4, HIGHPASS_CUTOFF,
        btype="highpass", fs=fs, output="sos"
    )
    return sosfiltfilt(sos, signal, axis=0)


def remove_powerline_noise(signal, fs):
    if NOTCH_FREQUENCY is None:
        return signal

    if not 0 < NOTCH_FREQUENCY < fs / 2:
        return signal

    b, a = iirnotch(NOTCH_FREQUENCY, Q=30, fs=fs)
    return filtfilt(b, a, signal, axis=0)


def remove_high_frequency_noise(signal, fs):
    sos = butter(
        4, LOWPASS_CUTOFF,
        btype="lowpass", fs=fs, output="sos"
    )
    return sosfiltfilt(sos, signal, axis=0)


def filter_ecg(signal, fs):
    signal = remove_baseline_wander(signal, fs)
    signal = remove_powerline_noise(signal, fs)
    signal = remove_high_frequency_noise(signal, fs)

    return signal.astype(np.float32)


# ---------------- STEP 4: ENCODE LABELS ----------------

def build_label_mapping(statements):
    diagnostic = statements[
        statements["diagnostic"].fillna(0).astype(int) == 1
    ]

    return diagnostic["diagnostic_class"].to_dict()


def encode_labels(scp_codes, code_to_class):
    """
    Convert SCP codes to five binary diagnostic superclass labels.
    A superclass is positive if at least one mapped code has
    positive confidence.
    """
    labels = np.zeros(len(CLASSES), dtype=np.float32)
    class_index = {name: i for i, name in enumerate(CLASSES)}

    for code, confidence in scp_codes.items():
        if confidence is None or confidence <= 0:
            continue

        category = code_to_class.get(code)

        if category in class_index:
            labels[class_index[category]] = 1.0

    return labels


# ---------------- STEP 5: SPLIT DATA ----------------

def split_metadata(metadata):
    train = metadata[metadata["strat_fold"].between(1, 8)].copy()
    val = metadata[metadata["strat_fold"] == 9].copy()
    test = metadata[metadata["strat_fold"] == 10].copy()

    return train, val, test


# ---------------- STEP 6: FIT NORMALIZATION ----------------

def fit_normalizer(train, data_dir):
    total = np.zeros(12, dtype=np.float64)
    total_sq = np.zeros(12, dtype=np.float64)
    count = 0

    for _, row in tqdm(
        train.iterrows(), total=len(train),
        desc="Fitting training normalization"
    ):
        signal, fs = load_ecg(row, data_dir, SAMPLING_RATE)
        signal = filter_ecg(signal, fs)

        total += signal.sum(axis=0, dtype=np.float64)
        total_sq += np.square(signal, dtype=np.float64).sum(axis=0)
        count += signal.shape[0]

    if count == 0:
        raise ValueError("No training signals found.")

    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


# ---------------- STEP 7: NORMALIZE ----------------

def normalize_ecg(signal, mean, std):
    return ((signal - mean) / std).astype(np.float32)


# ---------------- STEP 8: SAVE EACH SPLIT ----------------

def save_split(
    split_name, subset, data_dir, output_dir,
    mean, std, code_to_class
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Disk-backed arrays avoid holding the whole split in RAM.
    x_path = output_dir / f"X_{split_name}.npy"
    y_path = output_dir / f"y_{split_name}.npy"

    n = len(subset)
    if n == 0:
        raise ValueError(f"The {split_name} split is empty.")

    samples = int(SAMPLING_RATE * 10)
    X = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float32,
        shape=(n, samples, 12)
    )
    Y = np.lib.format.open_memmap(
        y_path, mode="w+", dtype=np.float32,
        shape=(n, len(CLASSES))
    )

    kept_rows = []
    index = 0

    try:
        for ecg_id, row in tqdm(
            subset.iterrows(), total=n,
            desc=f"Saving {split_name}"
        ):
            try:
                signal, fs = load_ecg(row, data_dir, SAMPLING_RATE)

                if signal.shape != (samples, 12):
                    print(f"Skipping ECG {ecg_id}: unexpected shape {signal.shape}")
                    continue

                signal = filter_ecg(signal, fs)
                signal = normalize_ecg(signal, mean, std)

                if not np.isfinite(signal).all():
                    print(f"Skipping ECG {ecg_id}: invalid filtered signal")
                    continue

                labels = encode_labels(row["scp_codes"], code_to_class)

                X[index] = signal
                Y[index] = labels
                kept_rows.append(ecg_id)
                index += 1

            except (OSError, ValueError, KeyError) as exc:
                print(f"Skipping ECG {ecg_id}: {exc}")

    finally:
        X.flush()
        Y.flush()
        del X, Y

    # If any records were skipped, trim the output files to the
    # actual number of successful records.
    if index == 0:
        x_path.unlink(missing_ok=True)
        y_path.unlink(missing_ok=True)
        raise ValueError(f"No valid ECG recordings in {split_name}.")

    if index < n:
        x_tmp = np.load(x_path, mmap_mode="r")
        y_tmp = np.load(y_path, mmap_mode="r")

        X_trim = np.lib.format.open_memmap(
            output_dir / f"X_{split_name}_trim.npy",
            mode="w+", dtype=np.float32,
            shape=(index, samples, 12)
        )
        Y_trim = np.lib.format.open_memmap(
            output_dir / f"y_{split_name}_trim.npy",
            mode="w+", dtype=np.float32,
            shape=(index, len(CLASSES))
        )

        X_trim[:] = x_tmp[:index]
        Y_trim[:] = y_tmp[:index]
        X_trim.flush()
        Y_trim.flush()

        del X_trim, Y_trim, x_tmp, y_tmp

        (output_dir / f"X_{split_name}_trim.npy").replace(x_path)
        (output_dir / f"y_{split_name}_trim.npy").replace(y_path)

    metadata_out = subset.loc[kept_rows].copy()
    metadata_out = metadata_out.drop(
        columns=["scp_codes"], errors="ignore"
    )
    metadata_out.to_csv(output_dir / f"{split_name}_metadata.csv")

    print(f"{split_name}: saved {index} recordings")


# ---------------- STEP 9: RUN PIPELINE ----------------

def main():
    metadata, statements = load_metadata(DATA_DIR)
    train, val, test = split_metadata(metadata)

    code_to_class = build_label_mapping(statements)

    mean, std = fit_normalizer(train, DATA_DIR)

    np.savez(
        OUTPUT_DIR / "preprocessing_stats.npz",
        mean=mean,
        std=std,
        classes=np.asarray(CLASSES),
        sampling_rate=SAMPLING_RATE,
        highpass_cutoff=HIGHPASS_CUTOFF,
        lowpass_cutoff=LOWPASS_CUTOFF,
    )

    save_split(
        "train", train, DATA_DIR, OUTPUT_DIR,
        mean, std, code_to_class
    )
    save_split(
        "val", val, DATA_DIR, OUTPUT_DIR,
        mean, std, code_to_class
    )
    save_split(
        "test", test, DATA_DIR, OUTPUT_DIR,
        mean, std, code_to_class
    )

    print(f"\nDataset saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
