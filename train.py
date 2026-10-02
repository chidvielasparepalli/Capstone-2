import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "breast_cancer_bd.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA}\n"
        "Place breast_cancer_bd.csv inside the data folder and run train.py again."
    )

df = pd.read_csv(DATA).replace("?", np.nan)

required = {"Class", "Sample code number", "Bare Nuclei"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

df["Bare Nuclei"] = pd.to_numeric(df["Bare Nuclei"], errors="coerce")
df = df.drop_duplicates().copy()

X = df.drop(columns=["Class", "Sample code number"])
y = df["Class"].astype(int)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = {
    "Random Forest": (
        ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__max_features": ["sqrt", "log2"],
        },
    ),
    "SVM": (
        ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("model", SVC(probability=True, random_state=42)),
        ]),
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"],
        },
    ),
    "Naive Bayes": (
        ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("model", GaussianNB()),
        ]),
        {
            "model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5],
        },
    ),
}

rows = []
fitted = {}

for name, (pipe, grid) in configs.items():
    gs = GridSearchCV(
        pipe,
        grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(Xtr, ytr)
    pred = gs.predict(Xte)

    rows.append({
        "Model": name,
        "CV Accuracy": gs.best_score_,
        "Test Accuracy": accuracy_score(yte, pred),
        "Class 4 Precision": precision_score(yte, pred, pos_label=4, zero_division=0),
        "Class 4 Recall": recall_score(yte, pred, pos_label=4, zero_division=0),
        "Class 4 F1": f1_score(yte, pred, pos_label=4, zero_division=0),
        "Best Params": json.dumps(gs.best_params_),
    })
    fitted[name] = gs.best_estimator_

results = pd.DataFrame(rows).sort_values(
    "Test Accuracy", ascending=False
).reset_index(drop=True)

best_name = results.iloc[0]["Model"]

with open(ART / "best_model.pkl", "wb") as f:
    pickle.dump(fitted[best_name], f)

results.to_csv(ART / "model_results.csv", index=False)

# Target distribution used by the Streamlit dashboard.
target_distribution = (
    y.value_counts()
    .sort_index()
    .rename_axis("Class")
    .reset_index(name="Count")
)
target_distribution.to_csv(ART / "target_distribution.csv", index=False)

# Random Forest feature importance used by the Streamlit dashboard.
rf_pipe = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )),
])
rf_pipe.fit(Xtr, ytr)

rf_model = rf_pipe.named_steps["model"]
feature_importance = (
    pd.DataFrame({
        "feature": X.columns,
        "importance": rf_model.feature_importances_,
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
feature_importance.to_csv(ART / "feature_importance.csv", index=False)

metadata = {
    "best_model": best_name,
    "dataset_shape": list(df.shape),
    "feature_count": int(X.shape[1]),
    "target_classes": sorted(y.unique().tolist()),
}
(ART / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(results.to_string(index=False))
print("\nGenerated artifacts:")
for path in sorted(ART.iterdir()):
    print(" -", path.name)
