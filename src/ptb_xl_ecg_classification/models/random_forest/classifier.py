from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from ptb_xl_ecg_classification.config import (
    CLASSES,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_RANDOM_STATE,
)


def tune_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Select the probability threshold with the best validation F1-score."""
    thresholds = np.unique(probabilities)
    candidates = np.concatenate(([0.0], thresholds, [1.0]))
    scores = [
        f1_score(
            y_true,
            probabilities >= threshold,
            zero_division=0,
        )
        for threshold in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
) -> RandomForestClassifier:
    """Train one balanced Random Forest for one diagnostic label."""
    model = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_FOREST_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(features, labels.astype(np.int8))
    return model


def predict_positive_probability(
    model: RandomForestClassifier,
    features: np.ndarray,
) -> np.ndarray:
    """Return the probability for the positive class."""
    return model.predict_proba(features)[:, 1]


def save_classifier(
    model: RandomForestClassifier,
    label: str,
    output_path: Path,
) -> None:
    """Save one trained classifier using the shared output directory."""
    models_path = output_path / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_path / f"random_forest_{label}.joblib")


def train_all_classifiers(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train, threshold, save, and predict all diagnostic classifiers."""
    validation_probabilities = np.zeros_like(
        validation_labels,
        dtype=np.float32,
    )
    test_probabilities = np.zeros_like(test_labels, dtype=np.float32)
    thresholds = np.zeros(len(CLASSES), dtype=np.float32)

    for index, label in enumerate(CLASSES):
        model = train_classifier(train_features, train_labels[:, index])
        validation_probabilities[:, index] = predict_positive_probability(
            model,
            validation_features,
        )
        test_probabilities[:, index] = predict_positive_probability(
            model,
            test_features,
        )
        thresholds[index] = tune_threshold(
            validation_labels[:, index],
            validation_probabilities[:, index],
        )
        save_classifier(model, label, output_path)
        print(f"{label}: validation threshold={thresholds[index]:.4f}")

    return validation_probabilities, test_probabilities, thresholds
