import numpy as np
import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from ptb_xl_ecg_classification.config import DATA_PATH, CLINICAL_COLS, OUTPUT_PATH


def run_training():
    features_path = OUTPUT_PATH / "svm/selected_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"File not found: {features_path}")

    selected_features_df = pd.read_csv(features_path)
    top_features = selected_features_df["feature"].tolist()

    X_train = np.load(DATA_PATH / "X_train.npy")
    y_train = np.load(DATA_PATH / "y_train.npy")
    train_meta = pd.read_csv(DATA_PATH / "train_metadata.csv", index_col="ecg_id")

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
    numeric_features = [c for c in ["age", "height", "weight", "sex"] if c in train_clin_df.columns]
    categorical_features = [c for c in ["heart_axis"] if c in train_clin_df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    X_train_clin = preprocessor.fit_transform(train_clin_df)
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features)) if categorical_features else []
    clinical_feature_names = numeric_features + cat_names

    pca = PCA(n_components=150, svd_solver='randomized', random_state=42)
    X_train_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], -1))

    X_train_comb = np.hstack([X_train_pca, X_train_clin])
    all_names = [f"PC_{i + 1}" for i in range(X_train_pca.shape[1])] + clinical_feature_names

    indices = [all_names.index(f) for f in top_features if f in all_names]
    X_train_filtered = X_train_comb[:, indices]

    model = OneVsRestClassifier(LinearSVC(C=1.0, random_state=42, max_iter=5000, class_weight='balanced'))
    model.fit(X_train_filtered, y_train)

    artifact = {
        "model": model,
        "pca": pca,
        "preprocessor": preprocessor,
        "top_features": top_features,
        "all_names": all_names
    }
    model_path = OUTPUT_PATH / "svm/svm_model.pkl"
    joblib.dump(artifact, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    run_training()