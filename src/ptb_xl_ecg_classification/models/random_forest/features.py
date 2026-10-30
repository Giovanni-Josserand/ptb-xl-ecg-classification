from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ptb_xl_ecg_classification.config import (
    DATA_PATH,
    RANDOM_FOREST_CHUNK_SIZE,
    RANDOM_FOREST_OUTPUT_PATH,
)


def extract_features(signals: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Convert normalized ECG tensors into fixed-length per-lead features."""
    if signals.ndim != 3 or signals.shape[2] != 12:
        raise ValueError(f"Expected (records, samples, 12), got {signals.shape}")

    feature_blocks = []
    feature_names = []
    statistics = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
        "rms": lambda values, axis: np.sqrt(
            np.mean(np.square(values), axis=axis)
        ),
    }

    for name, function in statistics.items():
        feature_blocks.append(function(signals, axis=1))
        feature_names.extend(
            f"{name}_lead_{lead + 1}" for lead in range(12)
        )

    feature_blocks.append(
        np.max(signals, axis=1) - np.min(signals, axis=1)
    )
    feature_names.extend(f"range_lead_{lead + 1}" for lead in range(12))

    differences = np.diff(signals, axis=1)
    feature_blocks.extend([
        np.mean(np.abs(differences), axis=1),
        np.std(differences, axis=1),
    ])
    feature_names.extend(
        f"{statistic}_difference_lead_{lead + 1}"
        for statistic in ("mean_abs", "std")
        for lead in range(12)
    )

    features = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced non-finite values.")

    return features, feature_names


def build_features(
    split: str,
    data_path: Path = DATA_PATH,
    output_path: Path = RANDOM_FOREST_OUTPUT_PATH,
) -> tuple[np.ndarray, list[str]]:
    """Build or load the cached feature matrix for one dataset split."""
    input_path = data_path / f"X_{split}.npy"
    cache_path = output_path / f"features_{split}.npy"
    signals = np.load(input_path, mmap_mode="r")

    if cache_path.exists():
        features = np.load(cache_path)
        _, feature_names = extract_features(np.asarray(signals[:1]))
        return features, feature_names

    feature_chunks = []
    feature_names = []
    for start in tqdm(
        range(0, len(signals), RANDOM_FOREST_CHUNK_SIZE),
        desc=f"Extracting {split} features",
    ):
        end = min(start + RANDOM_FOREST_CHUNK_SIZE, len(signals))
        features, feature_names = extract_features(
            np.asarray(signals[start:end])
        )
        feature_chunks.append(features)

    all_features = np.concatenate(feature_chunks, axis=0)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, all_features)
    return all_features, feature_names
