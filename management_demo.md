# 5-Minute Capstone 2 Demo

## 1. Problem

"Our project studies semiconductor manufacturing sensor data and predicts whether a production example will pass or fail."

## 2. Dataset

"We have 1,567 production examples and 590 sensor columns, plus a time column and the Pass/Fail target. The dataset has many missing sensor values, so data cleaning is an important part of the project."

## 3. Data analysis

Show **Data & EDA** and explain:

- missing values
- target imbalance
- one sensor distribution
- one sensor compared with Pass/Fail
- the correlation heatmap
- the monthly fail-rate chart

Always read the short **What this tells us** message below the chart.

## 4. Machine learning

"We compare Logistic Regression, SVM, and Balanced Random Forest. We use a stratified test split, five-fold cross-validation, median imputation, feature selection, and imbalance-aware training."

## 5. Prediction

Open **Prediction**.

Choose **First Pass example** and show the prediction.
Then choose **First Fail example** and show the prediction.

Explain:

"Because the real data has hundreds of sensor columns, it is better to predict using a complete real production example instead of asking the user to type hundreds of numbers."

## 6. Closing

"The goal is not only to train a model. The project also explains the data, handles missing values and class imbalance, compares different models, and provides an easy dashboard for inspection and prediction."
