"""Train models for semiconductor manufacturing yield prediction."""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "signal-data.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = "Pass/Fail"
TIME_COL = "Time"
MISSING_THRESHOLD = 0.50
TOP_K = 100


def make_pipeline(name: str, k: int) -> Pipeline:
    common = [
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_classif, k=k)),
    ]
    if name == "Logistic Regression":
        steps = common + [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000, solver="liblinear", random_state=RANDOM_STATE)),
        ]
    elif name == "SVM":
        steps = common + [
            ("scaler", StandardScaler()),
            ("model", SVC(C=1.0, kernel="linear", class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
        ]
    elif name == "Balanced Random Forest":
        steps = common + [
            ("model", BalancedRandomForestClassifier(n_estimators=350, random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    else:
        raise ValueError(name)
    return Pipeline(steps)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}. Put signal-data.csv inside the data folder."
        )

    df = pd.read_csv(DATA)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' is missing.")
    if TIME_COL not in df.columns:
        raise ValueError(f"Required time column '{TIME_COL}' is missing.")

    raw_shape = df.shape
    raw_missing_cells = int(df.isna().sum().sum())
    raw_duplicate_rows = int(df.duplicated().sum())

    sensor_columns = [c for c in df.columns if c not in [TIME_COL, TARGET]]
    X_raw = df[sensor_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[TARGET], errors="coerce").astype(int)

    missing_rate = X_raw.isna().mean()
    dropped_high_missing = missing_rate[missing_rate > MISSING_THRESHOLD].index.tolist()
    X_clean = X_raw.drop(columns=dropped_high_missing)
    dropped_constant = [c for c in X_clean.columns if X_clean[c].nunique(dropna=True) <= 1]
    X_clean = X_clean.drop(columns=dropped_constant)

    k = min(TOP_K, X_clean.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_names = ["Logistic Regression", "SVM", "Balanced Random Forest"]
    results = []
    fitted = {}

    for name in model_names:
        pipe = make_pipeline(name, k)
        cv_prob = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        cv_true = (y_train == 1).astype(int)

        best_threshold = 0.50
        best_balanced = -1.0
        for threshold in np.arange(0.20, 0.71, 0.01):
            cv_pred = (cv_prob >= threshold).astype(int)
            score = balanced_accuracy_score(cv_true, cv_pred)
            if score > best_balanced:
                best_balanced = score
                best_threshold = float(threshold)

        pipe.fit(X_train, y_train)
        test_prob = pipe.predict_proba(X_test)[:, 1]
        test_pred_binary = (test_prob >= best_threshold).astype(int)
        test_true_binary = (y_test == 1).astype(int)

        results.append({
            "Model": name,
            "CV Balanced Accuracy": float(best_balanced),
            "Test Accuracy": float(accuracy_score(test_true_binary, test_pred_binary)),
            "Balanced Accuracy": float(balanced_accuracy_score(test_true_binary, test_pred_binary)),
            "Fail Precision": float(precision_score(test_true_binary, test_pred_binary, zero_division=0)),
            "Fail Recall": float(recall_score(test_true_binary, test_pred_binary, zero_division=0)),
            "Fail F1": float(f1_score(test_true_binary, test_pred_binary, zero_division=0)),
            "ROC AUC": float(roc_auc_score(test_true_binary, test_prob)),
            "Probability Threshold": best_threshold,
        })
        fitted[name] = pipe

    results_df = pd.DataFrame(results).sort_values(
        ["Balanced Accuracy", "Fail Recall"], ascending=False
    ).reset_index(drop=True)
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]

    with open(ART / "best_model.pkl", "wb") as handle:
        pickle.dump(best_model, handle)
    results_df.to_csv(ART / "model_results.csv", index=False)

    target_distribution = pd.DataFrame({
        "Pass/Fail": [-1, 1],
        "Count": [int((y == -1).sum()), int((y == 1).sum())],
        "Label": ["Pass", "Fail"],
    })
    target_distribution.to_csv(ART / "target_distribution.csv", index=False)

    missing_summary = (
        pd.DataFrame({
            "Feature": X_raw.columns,
            "Missing Count": X_raw.isna().sum().values,
            "Missing Percent": X_raw.isna().mean().values * 100,
        })
        .sort_values("Missing Percent", ascending=False)
        .reset_index(drop=True)
    )
    missing_summary.to_csv(ART / "missing_summary.csv", index=False)

    statistics = X_clean.describe().T
    statistics["Missing Count"] = X_clean.isna().sum()
    statistics["Missing Percent"] = X_clean.isna().mean() * 100
    statistics.reset_index(names="Feature").to_csv(ART / "feature_statistics.csv", index=False)

    if best_name == "Balanced Random Forest":
        feature_model = best_model
    else:
        feature_model = make_pipeline("Balanced Random Forest", k)
        feature_model.fit(X_train, y_train)

    selector = feature_model.named_steps["select"]
    selected_names = X_clean.columns[selector.get_support()]
    tree_model = feature_model.named_steps["model"]
    importance = (
        pd.DataFrame({"Feature": selected_names, "Importance": tree_model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(ART / "feature_importance.csv", index=False)

    top_corr_features = importance.head(20)["Feature"].tolist()
    X_clean[top_corr_features].corr(method="spearman").to_csv(ART / "correlation_top20.csv")
    pd.DataFrame({"Feature": X_clean.columns}).to_csv(ART / "clean_features.csv", index=False)

    best_threshold = float(results_df.loc[results_df["Model"] == best_name, "Probability Threshold"].iloc[0])
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)
    confusion = confusion_matrix((y_test == 1).astype(int), test_pred).tolist()

    metadata = {
        "project": "Semiconductor Manufacturing Yield Prediction",
        "raw_dataset_shape": list(raw_shape),
        "raw_sensor_count": len(sensor_columns),
        "usable_sensor_count": int(X_clean.shape[1]),
        "selected_sensor_count": int(k),
        "missing_threshold_percent": MISSING_THRESHOLD * 100,
        "dropped_high_missing_count": len(dropped_high_missing),
        "dropped_constant_count": len(dropped_constant),
        "raw_missing_cells": raw_missing_cells,
        "raw_duplicate_rows": raw_duplicate_rows,
        "class_counts": {str(int(label)): int(count) for label, count in y.value_counts().to_dict().items()},
        "pass_label": -1,
        "fail_label": 1,
        "fail_rate_percent": float((y == 1).mean() * 100),
        "best_model": best_name,
        "best_probability_threshold": best_threshold,
        "test_confusion_matrix": confusion,
        "time_min": str(df[TIME_COL].min()),
        "time_max": str(df[TIME_COL].max()),
        "feature_names": X_clean.columns.tolist(),
        "dropped_high_missing_features": dropped_high_missing,
        "dropped_constant_features": dropped_constant,
    }
    (ART / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Raw dataset: {raw_shape[0]} rows x {raw_shape[1]} columns")
    print(f"Usable sensors after cleaning: {X_clean.shape[1]}")
    print(f"Dropped high-missing sensors: {len(dropped_high_missing)}")
    print(f"Dropped constant sensors: {len(dropped_constant)}")


if __name__ == "__main__":
    main()
