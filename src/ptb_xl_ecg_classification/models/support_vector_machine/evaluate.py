import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

from ptb_xl_ecg_classification.config import DATA_PATH, CLINICAL_COLS, CLASSES, OUTPUT_PATH


def run_evaluation():
    model_path = OUTPUT_PATH / "svm/svm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    artifact = joblib.load(model_path)
    model = artifact["model"]
    pca = artifact["pca"]
    preprocessor = artifact["preprocessor"]
    top_features = artifact["top_features"]
    all_names = artifact["all_names"]

    X_val = np.load(DATA_PATH / "X_val.npy")
    y_val = np.load(DATA_PATH / "y_val.npy")
    val_meta = pd.read_csv(DATA_PATH / "val_metadata.csv", index_col="ecg_id")

    X_test = np.load(DATA_PATH / "X_test.npy")
    y_test = np.load(DATA_PATH / "y_test.npy")
    test_meta = pd.read_csv(DATA_PATH / "test_metadata.csv", index_col="ecg_id")

    def prep_clinical_df(meta):
        df = meta[CLINICAL_COLS].copy()
        if "sex" in df.columns and df["sex"].dtype == object:
            df["sex"] = df["sex"].map({"M": 1, "F": 0, "Male": 1, "Female": 0}).fillna(0)
        for col in ["age", "height", "weight"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        if "heart_axis" in df.columns:
            df["heart_axis"] = df["heart_axis"].fillna("unknown").astype(str)
        return df

    def process_split(X, meta):
        clin_df = prep_clinical_df(meta)
        X_clin = preprocessor.transform(clin_df)
        X_pca = pca.transform(X.reshape(X.shape[0], -1))
        X_comb = np.hstack([X_pca, X_clin])
        indices = [all_names.index(f) for f in top_features if f in all_names]
        return X_comb[:, indices]

    X_val_filtered = process_split(X_val, val_meta)
    X_test_filtered = process_split(X_test, test_meta)

    def get_metrics(X, y):
        y_pred = model.predict(X)
        y_score = model.decision_function(X) if hasattr(model, "decision_function") else y_pred

        per_class = {}
        for i, cls in enumerate(CLASSES):
            p, r, f, s = precision_recall_fscore_support(y[:, i], y_pred[:, i], average='binary', zero_division=0)
            try:
                roc = roc_auc_score(y[:, i], y_score[:, i])
            except ValueError:
                roc = 0.0
            try:
                pr = average_precision_score(y[:, i], y_score[:, i])
            except ValueError:
                pr = 0.0

            per_class[cls] = {
                "support": int(s) if s is not None else 0,
                "precision": float(p) if p is not None else 0.0,
                "recall": float(r) if r is not None else 0.0,
                "f1": float(f) if f is not None else 0.0,
                "roc_auc": float(roc),
                "pr_auc": float(pr)
            }

        _, macro_p, macro_r, _ = precision_recall_fscore_support(y, y_pred, average='macro', zero_division=0)
        _, _, macro_f1, _ = precision_recall_fscore_support(y, y_pred, average='macro', zero_division=0)
        _, _, micro_f1, _ = precision_recall_fscore_support(y, y_pred, average='micro', zero_division=0)

        macro_pr_auc = float(np.mean([per_class[c]["pr_auc"] for c in CLASSES]))

        return {
            "per_class": per_class,
            "macro_f1": float(macro_f1) if macro_f1 is not None else 0.0,
            "micro_f1": float(micro_f1) if micro_f1 is not None else 0.0,
            "macro_precision": float(macro_p) if macro_p is not None else 0.0,
            "macro_recall": float(macro_r) if macro_r is not None else 0.0,
            "macro_pr_auc": macro_pr_auc
        }

    metrics = {
        "classes": CLASSES,
        "feature_count": len(top_features),
        "feature_names": top_features,
        "validation": get_metrics(X_val_filtered, y_val),
        "test": get_metrics(X_test_filtered, y_test)
    }

    svm_output_path = OUTPUT_PATH / "svm"
    svm_output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = svm_output_path / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    run_evaluation()