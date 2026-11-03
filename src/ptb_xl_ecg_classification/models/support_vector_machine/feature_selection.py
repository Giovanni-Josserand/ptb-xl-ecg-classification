import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance

from ptb_xl_ecg_classification.config import DATA_PATH, CLINICAL_COLS, OUTPUT_PATH


def run_feature_selection(top_k=1000, min_importance_threshold=0.01):
    X_train = np.load(DATA_PATH / "X_train.npy")
    y_train = np.load(DATA_PATH / "y_train.npy")
    X_val = np.load(DATA_PATH / "X_val.npy")
    y_val = np.load(DATA_PATH / "y_val.npy")

    train_meta = pd.read_csv(DATA_PATH / "train_metadata.csv", index_col="ecg_id")
    val_meta = pd.read_csv(DATA_PATH / "val_metadata.csv", index_col="ecg_id")

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

    train_clin_df = prep_clinical_df(train_meta)
    val_clin_df = prep_clinical_df(val_meta)

    numeric_features = [c for c in ["age", "height", "weight", "sex"] if c in train_clin_df.columns]
    categorical_features = [c for c in ["heart_axis"] if c in train_clin_df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    X_train_clin = preprocessor.fit_transform(train_clin_df)
    X_val_clin = preprocessor.transform(val_clin_df)

    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features)) if categorical_features else []
    clinical_feature_names = numeric_features + cat_names

    pca = PCA(n_components=150, svd_solver='randomized', random_state=42)
    X_train_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_val_pca = pca.transform(X_val.reshape(X_val.shape[0], -1))

    X_train_comb = np.hstack([X_train_pca, X_train_clin])
    X_val_comb = np.hstack([X_val_pca, X_val_clin])
    feature_names = [f"PC_{i + 1}" for i in range(X_train_pca.shape[1])] + clinical_feature_names

    model = OneVsRestClassifier(LinearSVC(C=1.0, random_state=42, max_iter=3000, class_weight='balanced'))
    model.fit(X_train_comb, y_train)

    subset = min(2000, X_val_comb.shape[0])
    result = permutation_importance(model, X_val_comb[:subset], y_val[:subset], n_repeats=3, random_state=42, n_jobs=-1)

    importances = np.maximum(result.importances_mean, 0)
    total = importances.sum()
    imp_pct = (importances / total * 100) if total > 0 else importances

    df_imp = pd.DataFrame({"feature": feature_names, "importance_pct": imp_pct})
    df_filtered = df_imp[df_imp["importance_pct"] > min_importance_threshold].copy()
    df_top = df_filtered.sort_values(by="importance_pct", ascending=False).head(top_k)

    svm_output_path = OUTPUT_PATH / "svm"
    svm_output_path.mkdir(parents=True, exist_ok=True)
    features_path = svm_output_path / "selected_features.csv"
    df_top.to_csv(features_path, index=False)
    print(f"Features saved to: {features_path}")


if __name__ == "__main__":
    run_feature_selection()