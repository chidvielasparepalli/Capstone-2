import json, pickle
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"
DATA=ROOT/"data"/"breast_cancer_bd.csv"

st.set_page_config(page_title="Breast Cancer ML Intelligence",page_icon="🧠",layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#07111d,#0e2234);color:#edf6ff}
.block-container{max-width:1400px;padding-top:1.5rem}
.hero{padding:1.5rem;border-radius:22px;background:rgba(15,38,57,.9);border:1px solid rgba(89,203,255,.2);margin-bottom:1rem}
.hero h1{margin:0}.hero p{color:#a9c5d8}
</style>""",unsafe_allow_html=True)

with open(ART/"best_model.pkl","rb") as f: model=pickle.load(f)
metrics=pd.read_csv(ART/"model_results.csv")
features=pd.read_csv(ART/"feature_importance.csv")
target=pd.read_csv(ART/"target_distribution.csv")
meta=json.loads((ART/"metadata.json").read_text())

st.markdown(f"""<div class="hero"><h1>🧠 Breast Cancer Classification</h1><p>Machine Learning Model Comparison • Feature Intelligence • Management Dashboard</p></div>""",unsafe_allow_html=True)
page=st.sidebar.radio("Sections",["Executive Overview","Data & EDA","Model Lab","Feature Intelligence","Prediction"])

if page=="Executive Overview":
    best=metrics.iloc[0]
    c=st.columns(5)
    items=[("Records",f"{meta.get('dataset_shape',[699,11])[0]:,}"),("Features","9"),("Best Model",best["Model"]),("Test Accuracy",f"{best['Test Accuracy']*100:.2f}%"),("Class 4 Recall",f"{best['Class 4 Recall']*100:.2f}%")]
    for col,(lab,val) in zip(c,items): col.metric(lab,val)
    st.subheader("Project objective")
    st.write("Build a supervised classifier, compare multiple algorithms, handle missing values and class imbalance, and identify the most informative measured attributes.")
    st.subheader("Model comparison")
    st.dataframe(metrics[["Model","CV Accuracy","Test Accuracy","Class 4 Precision","Class 4 Recall","Class 4 F1"]],use_container_width=True)
    fig,ax=plt.subplots(figsize=(9,4))
    chart=metrics.copy(); chart["Test Accuracy %"]=chart["Test Accuracy"]*100
    sns.barplot(data=chart,x="Model",y="Test Accuracy %",ax=ax)
    ax.set_ylim(90,100); ax.set_title("Test Accuracy")
    st.pyplot(fig,use_container_width=True)
    st.info("Academic demonstration only. This dashboard is not a medical diagnostic system.")

elif page=="Data & EDA":
    st.subheader("📊 Data & EDA")
    df=pd.read_csv(DATA).replace("?",pd.NA)
    c=st.columns(4)
    c[0].metric("Rows",len(df)); c[1].metric("Columns",len(df.columns))
    c[2].metric("Duplicate rows",int(df.duplicated().sum())); c[3].metric("Missing cells",int(df.isna().sum().sum()))
    st.dataframe(df.head(10),use_container_width=True)
    st.write("### Target distribution")
    td=target.copy(); td["Class"]=td["Class"].astype(str)
    fig,ax=plt.subplots(figsize=(7,4)); sns.barplot(data=td,x="Class",y="Count",ax=ax); ax.set_title("Class Distribution"); st.pyplot(fig,use_container_width=True)
    st.dataframe(df.describe(include="all").T,use_container_width=True)

elif page=="Model Lab":
    st.subheader("🧪 Model Lab")
    st.caption("Stratified 5-fold cross-validation + GridSearchCV + SMOTE inside each training pipeline.")
    st.dataframe(metrics,use_container_width=True)
    st.write(f"**{metrics.iloc[0]['Model']}** has the highest measured test accuracy on the project's fixed test split: **{metrics.iloc[0]['Test Accuracy']*100:.2f}%**.")
    st.write(f"Class 4 recall: **{metrics.iloc[0]['Class 4 Recall']*100:.2f}%**.")

elif page=="Feature Intelligence":
    st.subheader("🎯 Feature Intelligence")
    st.write("Random Forest feature importance ranks predictive attributes; it is not proof of causal relationships.")
    n=st.slider("Top features",5,min(9,len(features)),9)
    top=features.head(n)
    fig,ax=plt.subplots(figsize=(9,5)); sns.barplot(data=top.sort_values("importance"),x="importance",y="feature",ax=ax); ax.set_title("Most Informative Features"); st.pyplot(fig,use_container_width=True)
    st.dataframe(top,use_container_width=True)

elif page=="Prediction":
    st.subheader("🔮 Demo Prediction")
    st.caption("Enter the 9 measured attributes. The saved tuned model predicts Class 2 or Class 4.")
    feature_names=["Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"]
    with st.form("predict"):
        vals={}; cols=st.columns(3)
        for i,fname in enumerate(feature_names):
            with cols[i%3]: vals[fname]=st.number_input(fname,min_value=1.0,max_value=10.0,value=3.0,step=1.0)
        run=st.form_submit_button("Predict Class",type="primary")
    if run:
        row=pd.DataFrame([vals]); pred=model.predict(row)[0]
        st.success(f"Predicted class: {pred}")
        if hasattr(model,"predict_proba"):
            probs=model.predict_proba(row)[0]; st.write({str(cls):float(p) for cls,p in zip(model.classes_,probs)})
