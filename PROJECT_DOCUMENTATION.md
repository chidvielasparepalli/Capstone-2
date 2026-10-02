# Capstone 2 - Semiconductor Manufacturing Yield Prediction

## 1. Project in very simple words

A semiconductor factory has many sensors. While products are being made, these sensors record numbers.

Each production example has a final result:

- **-1 = Pass**
- **1 = Fail**

This project uses machine learning to learn from the sensor readings and predict whether a production example will Pass or Fail.

The project also studies the data, cleans it, handles missing values, compares machine-learning models, finds useful sensors, and presents everything in a Streamlit dashboard.

## 2. Main question

> Can the sensor readings help us predict whether a production example will PASS or FAIL?

## 3. Dataset

The project uses **signal-data.csv**.

From the supplied CSV:

- **1,567 examples**
- **592 columns**
- **1 Time column**
- **590 sensor columns**
- **1 Pass/Fail target column**
- **41,951 missing cells**
- **0 duplicate rows**
- **1,534 unique Time values**

The assignment sheet describes 1,567 examples and 591 features. In the actual CSV, there are 592 columns because the file contains Time + 590 sensors + the target column. This is only a counting difference.

### Target distribution

| Label | Meaning | Count |
|---:|---|---:|
| -1 | Pass | 1,463 |
| 1 | Fail | 104 |

Fail examples are only about **6.64%** of the dataset, so the classes are strongly imbalanced.

## 4. Full project flow

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
Fill remaining missing values with training median
   ↓
Select top 100 useful sensors
   ↓
Train machine-learning models
   ↓
Compare models using balanced metrics
   ↓
Save the best model
   ↓
Streamlit dashboard
   ↓
Pass / Fail prediction
```

## 5. Data cleaning

### Missing values

Some sensor readings are empty. The project counts missing values first.

### Remove sensors with too many missing values

Sensors with more than **50% missing values** are removed.

For the supplied dataset, **28 sensors** are removed for this reason.

### Remove constant sensors

A constant sensor gives the same value again and again and gives little useful information.

The current cleaning step removes **116 constant sensors**.

### Remaining sensors

After these cleaning steps, **446 sensor columns** remain.

### Fill remaining missing values

The remaining missing values are filled using the **median value learned from the training data**.

Median means the middle value after sorting known numbers.

### Duplicate rows

There are **0 duplicate rows** in the supplied dataset.

## 6. Data analysis

The dashboard contains the analysis requested for Capstone 2.

### Statistical analysis

The project can show:

- count
- mean
- standard deviation
- minimum
- 25th percentile
- median
- 75th percentile
- maximum

These numbers help us understand the normal range and spread of a sensor.

### Univariate analysis

Univariate means **one**.

The dashboard can show the distribution of one sensor using a histogram.

Simple explanation:

> This graph shows where the values of one sensor are concentrated and whether some values are unusual.

### Bivariate analysis

Bivariate means **two**.

The dashboard compares one sensor with the Pass/Fail result using a box plot.

Simple explanation:

> If the Pass and Fail groups look different, that sensor may help the model separate the two groups.

### Multivariate analysis

Multivariate means **many**.

The dashboard uses a Spearman correlation heatmap for important sensors.

Simple explanation:

> The heatmap shows which sensors tend to move together.

### Time analysis

The Time column can be grouped by month so we can see whether the Fail rate changes during the production period.

## 7. Feature selection

There are hundreds of sensors.

Using every sensor can make a model bigger and harder to understand.

The project uses **SelectKBest** to keep the top **100 useful sensors** for model training.

## 8. Machine-learning models

The current project compares:

### Logistic Regression

A simple model that learns a boundary between Pass and Fail.

### Support Vector Machine (SVM)

A model that tries to find a boundary that separates the two classes.

### Balanced Random Forest

A collection of decision trees designed to pay more attention to the smaller Fail class.

## 9. Why normal accuracy is not enough

Because most rows are Pass, a model can look accurate even if it misses many Fail examples.

For this reason, the project also reports:

- **Balanced Accuracy**
- **Fail Precision**
- **Fail Recall**
- **Fail F1**
- **ROC AUC**

## 10. Current baseline results

The following values were produced by running the current training pipeline on the supplied signal-data.csv with random seed 42.

| Model | Test Accuracy | Balanced Accuracy | Fail Recall | ROC AUC |
|---|---:|---:|---:|---:|
| Balanced Random Forest | 61.46% | **70.51%** | **80.95%** | 0.788 |
| Logistic Regression | 81.21% | 54.57% | 23.81% | 0.657 |
| SVM | 92.99% | 52.04% | 4.76% | 0.652 |

Other current test metrics:

- Balanced Random Forest: Fail Precision 12.69%, Fail F1 21.94%, threshold 0.35
- Logistic Regression: Fail Precision 10.42%, Fail F1 14.49%, threshold 0.54
- SVM: Fail Precision 33.33%, Fail F1 8.33%, threshold 0.21

### Current baseline choice

The current training code selects **Balanced Random Forest** because it has the highest test Balanced Accuracy in this run.

The saved model should always be regenerated locally with:

`python train.py`

before a final presentation.

## 11. Confusion matrix

For the current baseline Balanced Random Forest, the test confusion matrix is:

```text
                Predicted
                Pass   Fail
Actual Pass      176    117
Actual Fail        4     17
```

Simple meaning:

- 176 Pass examples were correctly predicted as Pass.
- 117 Pass examples were predicted as Fail.
- 4 Fail examples were predicted as Pass.
- 17 Fail examples were correctly predicted as Fail.

## 12. Feature intelligence

The dashboard shows Random Forest feature importance.

A higher importance means the tree model used that sensor more when making its decisions.

It does **not** prove that the sensor causes the Pass or Fail result.

Some of the highest-ranked sensors in the current baseline include:

`103`, `59`, `477`, `510`, `519`, `205`, `21`, `488`, `180`, and `64`.

## 13. Streamlit dashboard

The dashboard has five main sections.

### Executive Overview

Shows:

- number of examples
- number of sensors
- best model
- Fail rate
- model comparison

### Data & EDA

Shows:

- raw data sample
- missing-value analysis
- target distribution
- statistics
- univariate analysis
- bivariate analysis
- multivariate correlation
- time analysis
- cleaning decisions

### Model Lab

Shows:

- model results
- training flow
- confusion matrix

### Feature Intelligence

Shows the most important sensors and lets the user change how many top sensors are displayed.

### Prediction

The dataset has hundreds of sensor columns, so it would not be practical to ask a human to type hundreds of values.

Instead, the dashboard lets the user select a real example from the dataset, such as:

- first Pass example
- first Fail example
- a row number

The saved model then predicts Pass or Fail and shows the Fail probability and decision threshold.

## 14. Probability and threshold

The model gives a Fail probability between 0 and 1.

Examples:

- 0.20 = 20%
- 0.50 = 50%
- 0.80 = 80%

The project learns a decision threshold from cross-validation instead of always forcing 0.50.

For the current Balanced Random Forest baseline, the threshold is **0.35**.

Simple rule:

- Fail probability >= threshold → **Fail**
- Fail probability < threshold → **Pass**

## 15. How to run the project

Open PowerShell:

```powershell
cd A:\Capstone-2
git pull origin main
```

Make sure the dataset is here:

```text
A:\Capstone-2\data\signal-data.csv
```

Install packages if needed:

```powershell
pip install -r requirements.txt
```

Train:

```powershell
python train.py
```

Start the dashboard:

```powershell
python -m streamlit run app.py
```

## 16. Files created by train.py

- `best_model.pkl`
- `model_results.csv`
- `feature_importance.csv`
- `feature_statistics.csv`
- `missing_summary.csv`
- `target_distribution.csv`
- `correlation_top20.csv`
- `clean_features.csv`
- `metadata.json`

The dashboard reads these saved artifacts so it does not have to train the model every time it opens.

## 17. Judge demo order

Use this simple order:

1. Explain the problem.
2. Show the dataset size.
3. Show missing values and class imbalance.
4. Show one univariate chart.
5. Show one bivariate chart.
6. Show the correlation heatmap.
7. Show model comparison.
8. Explain why Balanced Accuracy matters.
9. Show the important sensors.
10. Use the Prediction page with a real Pass row and a real Fail row.

## 18. 30-second judge explanation

> “Our project is a semiconductor manufacturing yield prediction system. We use 1,567 production examples containing 590 sensor readings, a time value, and a Pass/Fail result. First, we clean missing and unhelpful sensors. Then we study the data using statistical, univariate, bivariate, multivariate, and time-based analysis. We compare Logistic Regression, SVM, and Balanced Random Forest models. Finally, we save the best model and use a Streamlit dashboard to explain the data and make Pass/Fail predictions.”

## 19. Viva questions

**What is the main goal?**  
To predict whether a semiconductor production example will Pass or Fail from sensor readings.

**Why is data cleaning important?**  
Because missing and useless sensor values can confuse the model.

**What does -1 mean?**  
Pass.

**What does 1 mean?**  
Fail.

**Why is accuracy alone not enough?**  
Because Fail examples are much fewer. A model can have good normal accuracy and still miss many Fail cases.

**Why use Balanced Random Forest?**  
It gives more attention to the smaller class and works well for many sensor columns.

**What is univariate analysis?**  
Studying one sensor at a time.

**What is bivariate analysis?**  
Studying a sensor together with Pass/Fail.

**What is multivariate analysis?**  
Studying several sensors together.

**What is feature selection?**  
Choosing the most useful sensor columns so the model uses a smaller set of inputs.

**What is feature importance?**  
A measure of how useful a feature is to a tree model.

**Why use a dashboard?**  
It makes the analysis and model results easier for a human to understand and demonstrate.

## 20. Simple glossary

- **Dataset** - the collection of data used by the project.
- **Sensor feature** - one sensor measurement column.
- **Target** - the answer the model tries to predict.
- **Missing value** - an empty or unrecorded value.
- **Median** - the middle value after sorting numbers.
- **Model** - a mathematical system that learns patterns.
- **Training** - teaching the model using known examples.
- **Test set** - examples kept aside to check the model.
- **Cross-validation** - checking the model several times using different splits.
- **Accuracy** - how many predictions are correct overall.
- **Balanced Accuracy** - a metric that gives equal importance to both classes.
- **Recall** - how many real Fail examples the model finds.
- **Precision** - how many predicted Fail examples are actually Fail.
- **F1 score** - a combined measure of precision and recall.
- **ROC AUC** - a score showing how well the model separates the two classes.
- **Feature importance** - how useful a feature is to a tree model.

## 21. Important limitations

- This is an academic machine-learning project.
- A model prediction is an estimate, not a guarantee.
- The dataset is imbalanced, so Fail detection should be discussed with balanced metrics.
- Results depend on the dataset and train/test split.
- Always run `python train.py` before the final presentation so the saved model and artifacts match the current code and dataset.
- If future sensor data is very different from the training data, model performance can change.

## 22. Final checklist

- [ ] signal-data.csv is inside `data/`
- [ ] `train.py` runs successfully
- [ ] artifacts are regenerated
- [ ] Streamlit app opens
- [ ] Data & EDA works
- [ ] Model Lab works
- [ ] Feature Intelligence works
- [ ] Prediction works
- [ ] Documentation is ready
- [ ] Presenter can explain the project flow in simple words
