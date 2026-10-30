from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


DATASET_CLEAN_PATH = Path("data/ptb-xl/ptbxl_clean")
OUTPUT_PATH = Path("data/ptb-xl/random_forest")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
RANDOM_STATE = 42
N_ESTIMATORS = 300
CHUNK_SIZE = 512


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
        "rms": lambda values, axis: np.sqrt(np.mean(np.square(values), axis=axis)),
    }
    for name, function in statistics.items():
        feature_blocks.append(function(signals, axis=1))
        feature_names.extend(f"{name}_lead_{lead + 1}" for lead in range(12))

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


def build_features(split: str) -> tuple[np.ndarray, list[str]]:
    input_path = DATASET_CLEAN_PATH / f"X_{split}.npy"
    cache_path = OUTPUT_PATH / f"features_{split}.npy"
    signals = np.load(input_path, mmap_mode="r")

    if cache_path.exists():
        features = np.load(cache_path)
        _, feature_names = extract_features(np.asarray(signals[:1]))
        return features, feature_names

    feature_chunks = []
    feature_names = []
    for start in tqdm(
        range(0, len(signals), CHUNK_SIZE),
        desc=f"Extracting {split} features",
    ):
        end = min(start + CHUNK_SIZE, len(signals))
        features, feature_names = extract_features(np.asarray(signals[start:end]))
        feature_chunks.append(features)

    all_features = np.concatenate(feature_chunks, axis=0)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, all_features)
    return all_features, feature_names


def tune_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.unique(probabilities)
    candidates = np.concatenate(([0.0], thresholds, [1.0]))
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    predictions = probabilities >= thresholds
    per_class = {}
    for index, name in enumerate(CLASSES):
        per_class[name] = {
            "support": int(y_true[:, index].sum()),
            "precision": float(
                precision_score(y_true[:, index], predictions[:, index], zero_division=0)
            ),
            "recall": float(
                recall_score(y_true[:, index], predictions[:, index], zero_division=0)
            ),
            "f1": float(
                f1_score(y_true[:, index], predictions[:, index], zero_division=0)
            ),
            "roc_auc": safe_roc_auc(y_true[:, index], probabilities[:, index]),
            "pr_auc": float(
                average_precision_score(y_true[:, index], probabilities[:, index])
            ),
        }

    return {
        "per_class": per_class,
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_pr_auc": float(
            np.mean([
                average_precision_score(y_true[:, index], probabilities[:, index])
                for index in range(len(CLASSES))
            ])
        ),
    }


def main() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    y_train = np.load(DATASET_CLEAN_PATH / "y_train.npy")
    y_val = np.load(DATASET_CLEAN_PATH / "y_val.npy")
    y_test = np.load(DATASET_CLEAN_PATH / "y_test.npy")

    x_train, feature_names = build_features("train")
    x_val, _ = build_features("val")
    x_test, _ = build_features("test")

    validation_probabilities = np.zeros_like(y_val, dtype=np.float32)
    test_probabilities = np.zeros_like(y_test, dtype=np.float32)
    thresholds = np.zeros(len(CLASSES), dtype=np.float32)

    models_path = OUTPUT_PATH / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(CLASSES):
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(x_train, y_train[:, index].astype(np.int8))
        validation_probabilities[:, index] = model.predict_proba(x_val)[:, 1]
        test_probabilities[:, index] = model.predict_proba(x_test)[:, 1]
        thresholds[index] = tune_threshold(
            y_val[:, index], validation_probabilities[:, index]
        )
        joblib.dump(model, models_path / f"random_forest_{name}.joblib")
        print(f"{name}: validation threshold={thresholds[index]:.4f}")

    validation_metrics = evaluate(y_val, validation_probabilities, thresholds)
    test_metrics = evaluate(y_test, test_probabilities, thresholds)
    results = {
        "classes": CLASSES,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "n_estimators": N_ESTIMATORS,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "thresholds": dict(zip(CLASSES, thresholds.astype(float))),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    (OUTPUT_PATH / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"validation": validation_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
