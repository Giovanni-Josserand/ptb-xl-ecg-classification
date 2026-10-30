from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ptb_xl_ecg_classification.config import CLASSES


def safe_roc_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Calculate per-label and aggregate multi-label metrics."""
    predictions = probabilities >= thresholds
    per_class = {}

    for index, label in enumerate(CLASSES):
        per_class[label] = {
            "support": int(y_true[:, index].sum()),
            "precision": float(
                precision_score(
                    y_true[:, index],
                    predictions[:, index],
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true[:, index],
                    predictions[:, index],
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true[:, index],
                    predictions[:, index],
                    zero_division=0,
                )
            ),
            "roc_auc": safe_roc_auc(
                y_true[:, index],
                probabilities[:, index],
            ),
            "pr_auc": float(
                average_precision_score(
                    y_true[:, index],
                    probabilities[:, index],
                )
            ),
        }

    return {
        "per_class": per_class,
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_pr_auc": float(
            np.mean([
                average_precision_score(
                    y_true[:, index],
                    probabilities[:, index],
                )
                for index in range(len(CLASSES))
            ])
        ),
    }


def save_metrics(results: dict, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metrics.json").write_text(
        json.dumps(results, indent=2)
    )
