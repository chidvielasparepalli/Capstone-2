import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "signal-data.csv"
ART = ROOT / "artifacts"

st.set_page_config(
    page_title="Semiconductor Yield Intelligence",
    page_icon="🏭",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg,#06131d,#0f2433); color:#edf6ff; }
    .block-container { max-width: 1450px; padding-top: 1.3rem; }
    .hero { padding: 1.4rem 1.6rem; border-radius: 22px; background: rgba(13,35,50,.94); border: 1px solid rgba(88,205,255,.22); margin-bottom: 1rem; }
    .hero h1 { margin: 0; }
    .hero p { color:#a9c5d8; margin:.35rem 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

if not DATA.exists():
    st.error("signal-data.csv was not found. Put it inside the data folder.")
    st.stop()

with open(ART / "best_model.pkl", "rb") as handle:
    model = pickle.load(handle)
metrics = pd.read_csv(ART / "model_results.csv")
importance = pd.read_csv(ART / "feature_importance.csv")
target = pd.read_csv(ART / "target_distribution.csv")
missing = pd.read_csv(ART / "missing_summary.csv")
meta = json.loads((ART / "metadata.json").read_text(encoding="utf-8"))
df = pd.read_csv(DATA)

st.markdown(
    """
    <div class="hero">
      <h1>🏭 Semiconductor Manufacturing Yield Prediction</h1>
      <p>Sensor analysis • Machine learning • Yield risk prediction • Explainable dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Sections",
    ["Executive Overview", "Data & EDA", "Model Lab", "Feature Intelligence", "Prediction"],
)


def observation(text: str):
    st.info("**What this tells us:** " + text)


def plot_target():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(target["Label"], target["Count"])
    ax.set_title("Pass vs Fail Count")
    ax.set_xlabel("Yield Result")
    ax.set_ylabel("Number of Examples")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


if page == "Executive Overview":
    best = metrics.iloc[0]
    c = st.columns(5)
    items = [
        ("Examples", f"{meta['raw_dataset_shape'][0]:,}"),
        ("Raw Sensors", f"{meta['raw_sensor_count']:,}"),
        ("Usable Sensors", f"{meta['usable_sensor_count']:,}"),
        ("Best Model", best["Model"]),
        ("Fail Rate", f"{meta['fail_rate_percent']:.2f}%"),
    ]
    for col, (label, value) in zip(c, items):
        col.metric(label, value)

    st.subheader("🎯 Project objective")
    st.write(
        "Use machine learning to study factory sensor readings and predict whether a production example is likely to pass or fail. "
        "The dashboard also explains the data, missing values, important sensors, and model quality."
    )

    st.subheader("Model comparison")
    st.dataframe(metrics, width="stretch")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(metrics["Model"], metrics["Balanced Accuracy"] * 100)
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    observation(
        "Because only a small part of the dataset is Fail, normal accuracy alone can look better than it really is. "
        "Balanced accuracy and Fail Recall are shown so we can also see how well the model finds the minority Fail cases."
    )
    st.warning("Academic project only. The prediction is a machine-learning estimate, not a replacement for engineering quality control.")


elif page == "Data & EDA":
    st.subheader("📊 Data & Exploratory Data Analysis")
    c = st.columns(4)
    c[0].metric("Rows", len(df))
    c[1].metric("Columns", len(df.columns))
    c[2].metric("Missing Cells", int(df.isna().sum().sum()))
    c[3].metric("Duplicate Rows", int(df.duplicated().sum()))

    with st.expander("1. Raw data sample", expanded=True):
        st.dataframe(df.head(8), width="stretch")
        observation(
            "The dataset contains 1,567 production examples, one time column, 590 sensor columns, and one Pass/Fail target column."
        )

    with st.expander("2. Missing-value analysis", expanded=True):
        st.dataframe(missing.head(20), width="stretch")
        observation(
            "Many sensor readings are missing. The training pipeline removes sensors with more than 50% missing values and fills the remaining missing values with the median learned from the training data."
        )

    with st.expander("3. Target analysis", expanded=True):
        plot_target()
        observation(
            "The Fail class is much smaller than the Pass class. This class imbalance is why the project uses balanced models and checks Fail Recall and Balanced Accuracy."
        )

    with st.expander("4. Univariate analysis", expanded=True):
        top_feature = st.selectbox("Choose a sensor", importance["Feature"].head(20).tolist())
        values = pd.to_numeric(df[top_feature], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(values, bins=30)
        ax.set_title(f"Distribution of Sensor {top_feature}")
        ax.set_xlabel("Sensor value")
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        st.dataframe(pd.DataFrame(values.describe()).T, width="stretch")
        observation(
            f"This chart shows the spread of sensor {top_feature}. A histogram helps us see common values, unusual values, and whether the readings are tightly grouped or widely spread."
        )

    with st.expander("5. Bivariate analysis", expanded=True):
        sensor = st.selectbox("Sensor to compare with yield result", importance["Feature"].head(15).tolist(), key="biv")
        plot_df = pd.DataFrame({"value": pd.to_numeric(df[sensor], errors="coerce"), "label": df["Pass/Fail"].map({-1: "Pass", 1: "Fail"})}).dropna()
        groups = [plot_df.loc[plot_df["label"] == "Pass", "value"], plot_df.loc[plot_df["label"] == "Fail", "value"]]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(groups, tick_labels=["Pass", "Fail"], showfliers=False)
        ax.set_title(f"Sensor {sensor} vs Yield Result")
        ax.set_ylabel("Sensor value")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        observation(
            "This comparison shows whether the sensor values look different between Pass and Fail examples. A visible separation can mean the sensor may help the model distinguish yield outcomes."
        )

    with st.expander("6. Multivariate analysis", expanded=True):
        top_features = importance.head(12)["Feature"].tolist()
        corr = df[top_features].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(corr.fillna(0).values, aspect="auto")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(top_features)), top_features, rotation=90, fontsize=7)
        ax.set_yticks(range(len(top_features)), top_features, fontsize=7)
        ax.set_title("Spearman Correlation of Top Sensors")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        observation(
            "This heatmap looks at many sensors together. Strong positive or negative values mean two sensors tend to move together. Correlated sensors may contain similar information."
        )

    with st.expander("7. Time analysis", expanded=False):
        time_df = df[["Time", "Pass/Fail"]].copy()
        time_df["Time"] = pd.to_datetime(time_df["Time"], errors="coerce")
        time_df["Month"] = time_df["Time"].dt.to_period("M").astype(str)
        monthly = time_df.assign(Fail=(time_df["Pass/Fail"] == 1).astype(int)).groupby("Month")["Fail"].mean() * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(monthly.index, monthly.values, marker="o")
        ax.set_title("Monthly Fail Rate")
        ax.set_ylabel("Fail Rate (%)")
        ax.set_xlabel("Month")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        observation("The time view helps us see whether the share of failures changes across the production period.")

    with st.expander("8. Cleaning decisions", expanded=False):
        st.write(f"Sensors with more than {meta['missing_threshold_percent']:.0f}% missing values removed: {meta['dropped_high_missing_count']}")
        st.write(f"Constant sensors removed: {meta['dropped_constant_count']}")
        st.write(f"Sensors kept for modeling: {meta['usable_sensor_count']}")
        observation("The cleaning step removes sensors that have too little useful information, while remaining missing values are safely filled during model training.")


elif page == "Model Lab":
    st.subheader("🧪 Model Lab")
    st.caption("Five-fold stratified cross-validation, missing-value imputation, top-feature selection, and imbalance-aware models.")
    st.dataframe(metrics, width="stretch")
    best = metrics.iloc[0]
    st.write(
        f"**Best model:** {best['Model']} with a test balanced accuracy of **{best['Balanced Accuracy']*100:.2f}%** "
        f"and Fail Recall of **{best['Fail Recall']*100:.2f}%**."
    )
    st.write("### How the training flow works")
    st.code("""Raw sensor data
  ↓
Remove high-missing and constant sensors
  ↓
Fill remaining missing values with training median
  ↓
Select top 100 useful sensors
  ↓
Train balanced ML model
  ↓
Check test performance
  ↓
Save best model""")
    st.subheader("Confusion matrix")
    cm = np.array(meta["test_confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], ["Pass", "Fail"])
    ax.set_yticks([0, 1], ["Pass", "Fail"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Test Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    observation("The confusion matrix shows how many Pass and Fail examples were classified correctly or incorrectly on the test data.")


elif page == "Feature Intelligence":
    st.subheader("🎯 Feature Intelligence")
    n = st.slider("How many important sensors to show", 5, min(30, len(importance)), 15)
    top = importance.head(n).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["Feature"], top["Importance"])
    ax.set_title("Most Important Sensors")
    ax.set_xlabel("Random Forest importance")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.dataframe(importance.head(n), width="stretch")
    observation("A higher importance means the tree model used that sensor more often to separate outcomes. Importance shows usefulness for prediction, not a cause-and-effect relationship.")


elif page == "Prediction":
    st.subheader("🔮 Yield Prediction")
    st.write("This project uses 590 raw sensor columns, so manual entry of hundreds of values would not be practical. Instead, choose a real dataset example and let the saved model predict it.")
    pass_rows = df.index[df["Pass/Fail"] == -1].tolist()
    fail_rows = df.index[df["Pass/Fail"] == 1].tolist()
    choice = st.selectbox("Choose an example", ["First Pass example", "First Fail example", "Choose row number"])
    if choice == "First Pass example":
        row_index = pass_rows[0]
    elif choice == "First Fail example":
        row_index = fail_rows[0]
    else:
        row_index = st.number_input("Row number", min_value=0, max_value=len(df)-1, value=0, step=1)

    row = df.iloc[[int(row_index)]]
    model_features = meta["feature_names"]
    row_features = row[model_features].apply(pd.to_numeric, errors="coerce")
    prob_fail = float(model.predict_proba(row_features)[:, 1][0])
    threshold = float(meta["best_probability_threshold"])
    pred = 1 if prob_fail >= threshold else -1
    actual = int(row["Pass/Fail"].iloc[0])

    c = st.columns(4)
    c[0].metric("Selected Row", int(row_index))
    c[1].metric("Actual Result", "Fail" if actual == 1 else "Pass")
    c[2].metric("Predicted Result", "Fail" if pred == 1 else "Pass")
    c[3].metric("Fail Probability", f"{prob_fail*100:.2f}%")

    st.progress(min(prob_fail, 1.0), text=f"Model fail probability: {prob_fail*100:.2f}%")
    if pred == 1:
        st.error("Prediction: FAIL")
    else:
        st.success("Prediction: PASS")

    st.write(f"Decision threshold used by the trained model: **{threshold:.2f}**")
    st.dataframe(row[["Time", "Pass/Fail"]], width="stretch")
    observation("The dashboard feeds the selected row's sensor readings to the saved model. The model returns a probability for Fail, and the probability is compared with the learned decision threshold.")
