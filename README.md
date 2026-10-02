# Breast Cancer Classification Capstone

Management-ready machine learning project built from the supplied breast_cancer_bd.csv.

## Models
- Random Forest
- SVM
- Naive Bayes

## Workflow
EDA, cleansing, SMOTE, standardization, stratified 5-fold cross-validation, GridSearchCV, model comparison, feature importance, and model saving.

## Dataset note
The raw CSV is intentionally not committed to this public repository. Place `breast_cancer_bd.csv` in `data/` locally before retraining.

## Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Academic demonstration only; not a clinical diagnostic system.
