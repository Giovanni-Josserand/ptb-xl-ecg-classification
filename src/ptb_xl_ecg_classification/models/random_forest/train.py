from __future__ import annotations

import json

import numpy as np

from ptb_xl_ecg_classification.config import (
    CLASSES,
    DATA_PATH,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_OUTPUT_PATH,
    RANDOM_FOREST_RANDOM_STATE,
)
from ptb_xl_ecg_classification.models.random_forest.classifier import (
    train_all_classifiers,
)
from ptb_xl_ecg_classification.models.random_forest.evaluate import (
    evaluate,
    save_metrics,
)
from ptb_xl_ecg_classification.models.random_forest.features import (
    build_features,
)


def main() -> None:
    RANDOM_FOREST_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    y_train = np.load(DATA_PATH / "y_train.npy")
    y_val = np.load(DATA_PATH / "y_val.npy")
    y_test = np.load(DATA_PATH / "y_test.npy")

    x_train, feature_names = build_features("train")
    x_val, _ = build_features("val")
    x_test, _ = build_features("test")

    (
        validation_probabilities,
        test_probabilities,
        thresholds,
    ) = train_all_classifiers(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        RANDOM_FOREST_OUTPUT_PATH,
    )

    validation_metrics = evaluate(
        y_val,
        validation_probabilities,
        thresholds,
    )
    test_metrics = evaluate(
        y_test,
        test_probabilities,
        thresholds,
    )
    results = {
        "classes": CLASSES,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "n_estimators": RANDOM_FOREST_N_ESTIMATORS,
        "class_weight": "balanced",
        "random_state": RANDOM_FOREST_RANDOM_STATE,
        "thresholds": dict(zip(CLASSES, thresholds.astype(float))),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    save_metrics(results, RANDOM_FOREST_OUTPUT_PATH)
    print(json.dumps({"validation": validation_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
