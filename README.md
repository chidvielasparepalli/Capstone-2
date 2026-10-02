# 🏭 Semiconductor Manufacturing Yield Prediction — Capstone 2

## 1. What is this project?

This project uses machine learning to study sensor readings from a semiconductor manufacturing process.

The main question is simple:

> **Can the sensor readings help us predict whether a production example will PASS or FAIL?**

The project includes data cleaning, statistical analysis, visualisation, machine learning, model comparison, feature importance, and an interactive Streamlit dashboard.

## 2. Dataset

The supplied file is `signal-data.csv`.

- Examples: **1,567**
- Raw sensor columns: **590**
- Time column: `Time`
- Target column: `Pass/Fail`
- Pass label: `-1`
- Fail label: `1`
- Missing cells: **41,951**
- Duplicate rows: **0**

The classes are imbalanced: most examples are Pass and a smaller group are Fail. Because of this, the project reports **Balanced Accuracy, Fail Precision, Fail Recall, Fail F1, and ROC AUC**, not accuracy alone.

## 3. ML workflow

```text
Signal data
   ↓
Explore the data
   ↓
Check missing values and duplicates
   ↓
Remove sensors with >50% missing values
   ↓
Remove constant sensors
   ↓
Fill remaining missing values using the training median
   ↓
Select the top 100 useful sensors
   ↓
Train 3 models
   ↓
Compare their results
   ↓
Save the best model
   ↓
Use the model in the Streamlit dashboard
```

## 4. Models

The project compares:

- Logistic Regression
- Support Vector Machine (SVM)
- Balanced Random Forest

Balanced Random Forest is designed to give more attention to the smaller Fail class.

## 5. Analysis required by Capstone 2

The dashboard contains:

- Data import and exploration
- Missing-value analysis
- Data cleaning decisions
- Statistical summaries
- Univariate analysis
- Bivariate analysis
- Multivariate correlation analysis
- Time-based analysis
- Visualisation
- Model comparison
- Confusion matrix
- Feature importance
- Interactive prediction
- Simple comments after every major analysis

## 6. How to run

Open PowerShell in the project folder:

```powershell
pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

## 7. Project structure

```text
Capstone-2/
├── app.py
├── train.py
├── requirements.txt
├── management_demo.md
├── data/
│   └── signal-data.csv
├── artifacts/
│   ├── best_model.pkl
│   ├── model_results.csv
│   ├── feature_importance.csv
│   ├── feature_statistics.csv
│   ├── missing_summary.csv
│   ├── target_distribution.csv
│   ├── correlation_top20.csv
│   ├── clean_features.csv
│   └── metadata.json
└── notebooks/
```

## 8. Important note

This is an academic machine-learning project for semiconductor manufacturing yield analysis. A model prediction should be treated as decision support, not as proof that a real production process is safe or unsafe.
