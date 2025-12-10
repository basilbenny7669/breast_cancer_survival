import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# -----------------------------
# 1. Paths and config
# -----------------------------
DATA_PATH = Path("data") / "metabric_rna_mutation.csv" # <-- change name if needed
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "breast_cancer_survival_rf.joblib"


def load_data():
   
    df = pd.read_csv(DATA_PATH)

    # Show basic info once
    print("Data shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist()[:40], "...")
    return df


def build_features_and_target(df: pd.DataFrame):
   

    # ---- Target ----
    target_col = "overall_survival"

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    # Make sure target is 0/1 integer (if it isn't already)
    df[target_col] = df[target_col].astype(int)

 
    clinical_features = [
        "age_at_diagnosis",
        "tumor_size",
        "lymph_nodes_examined_positive",
        "type_of_breast_surgery",
        "cancer_type_detailed",
        "cellularity",
        "er_status",
        "her2_status",
        "hormone_therapy",
        "chemotherapy",
        "neoplasm_histologic_grade",
    ]

    # Keep only required columns
    missing = [c for c in clinical_features + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"The following required columns are missing from the CSV: {missing}")

    df_model = df[clinical_features + [target_col]].copy()

    # Optional: standardise Yes/No style if present as text
    yes_no_cols = ["hormone_therapy", "chemotherapy"]
    for col in yes_no_cols:
        if df_model[col].dtype == "O": # object / string
            df_model[col] = df_model[col].str.strip().str.upper()
            df_model[col] = df_model[col].replace(
                {
                    "YES": "Yes",
                    "NO": "No",
                    "Y": "Yes",
                    "N": "No",
                }
            )

    # ---- Define categorical vs numeric explicitly (IMPORTANT) ----
    categorical_features = [
        "type_of_breast_surgery",
        "cancer_type_detailed",
        "cellularity",
        "er_status",
        "her2_status",
        "hormone_therapy", # force categorical, even if 0/1 in CSV
        "chemotherapy", # force categorical, even if 0/1 in CSV
    ]

    numeric_features = [
        "age_at_diagnosis",
        "tumor_size",
        "lymph_nodes_examined_positive",
        "neoplasm_histologic_grade",
    ]

    X = df_model[clinical_features]
    y = df_model[target_col]

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("\nTarget value counts:\n", y.value_counts())

    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features):
    """Build the preprocessing + model pipeline."""

    # Numeric: impute median
    numeric_transformer = SimpleImputer(strategy="median")

    # Categorical: impute most frequent + one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    return model


def main():
    # 1. Load data
    df = load_data()

    # 2. Build X, y
    X, y, numeric_features, categorical_features = build_features_and_target(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

    # 4. Build pipeline
    model = build_pipeline(numeric_features, categorical_features)

    # 5. Fit model
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # 6. Evaluate
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # 7. Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}\n")

    # Optional: show which columns the preprocessor thinks are numeric/categorical
    preprocessor = model.named_steps["preprocessor"]
    print("Preprocessor transformers:")
    for name, transformer, cols in preprocessor.transformers_:
        print(f" - {name}: {list(cols)}")


if __name__ == "__main__":
    main()