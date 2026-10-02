# 🧠 Breast Cancer Classification — Capstone 2

A management-ready machine learning capstone built from the supplied breast cancer dataset.

## Project Objective

Build and compare supervised learning models, handle missing values and class imbalance, identify important attributes, and expose the trained model through a Streamlit dashboard.

## ML Workflow

EDA → Data Cleaning → Visualization → Train/Test Split → Standardization → SMOTE → 5-Fold Cross-Validation → GridSearchCV → Model Comparison → Feature Importance → Model Saving → Dashboard

### Models
- Random Forest
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

### Actual test results
| Model | Test Accuracy |
|---|---:|
| Naive Bayes | **97.12%** |
| Random Forest | **96.40%** |
| SVM | **96.40%** |

The current best model by the fixed test split is Naive Bayes (97.12%).

## Dataset

The supplied file contains 699 records and 11 original columns.

- Target: `Class`
- Identifier: `Sample code number` — excluded from model training
- Missing values: `?` in `Bare Nuclei`
- Missing numerical values are median-imputed inside the model pipeline.
- SMOTE is applied only within training folds.

> The raw CSV is intentionally **not committed** to this public repository. Place `breast_cancer_bd.csv` in `data/` locally before retraining.

## Dashboard

The Streamlit dashboard provides:
- Executive Overview
- Data & EDA
- Model Lab
- Feature Intelligence
- Interactive Prediction

## Run Locally

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Project Structure

```text
Capstone-2/
├── app.py
├── train.py
├── requirements.txt
├── management_demo.md
├── data/
├── artifacts/
└── notebooks/
```

## Management Demo

See `management_demo.md` for the recommended 5-minute presentation flow.

## Academic Notice

This project is an academic machine-learning demonstration. It is **not a clinical diagnostic system** and should not be used for medical decisions.
