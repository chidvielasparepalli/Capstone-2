import json, pickle
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"breast_cancer_bd.csv"
ART=ROOT/"artifacts"; ART.mkdir(exist_ok=True)
df=pd.read_csv(DATA).replace("?", np.nan)
df["Bare Nuclei"]=pd.to_numeric(df["Bare Nuclei"],errors="coerce")
df=df.drop_duplicates().copy()
X=df.drop(columns=["Class","Sample code number"])
y=df["Class"].astype(int)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,stratify=y,random_state=42)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
configs={
"Random Forest":(
ImbPipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("smote",SMOTE(random_state=42)),("model",RandomForestClassifier(random_state=42,n_jobs=-1))]),
{"model__n_estimators":[200,400],"model__max_depth":[None,10,20],"model__min_samples_split":[2,5],"model__max_features":["sqrt","log2"]}),
"SVM":(
ImbPipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("smote",SMOTE(random_state=42)),("model",SVC(probability=True,random_state=42))]),
{"model__C":[.1,1,10],"model__kernel":["rbf","linear"],"model__gamma":["scale","auto"]}),
"Naive Bayes":(
ImbPipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("smote",SMOTE(random_state=42)),("model",GaussianNB())]),
{"model__var_smoothing":[1e-11,1e-9,1e-7,1e-5]})
}
rows=[]; fitted={}
for name,(pipe,grid) in configs.items():
    gs=GridSearchCV(pipe,grid,cv=cv,scoring="accuracy",n_jobs=-1,refit=True)
    gs.fit(Xtr,ytr); p=gs.predict(Xte)
    rows.append({"Model":name,"CV Accuracy":gs.best_score_,"Test Accuracy":accuracy_score(yte,p),"Class 4 Precision":precision_score(yte,p,pos_label=4,zero_division=0),"Class 4 Recall":recall_score(yte,p,pos_label=4,zero_division=0),"Class 4 F1":f1_score(yte,p,pos_label=4,zero_division=0),"Best Params":json.dumps(gs.best_params_)})
    fitted[name]=gs.best_estimator_
out=pd.DataFrame(rows).sort_values("Test Accuracy",ascending=False)
best=out.iloc[0]["Model"]
with open(ART/"best_model.pkl","wb") as f: pickle.dump(fitted[best],f)
out.to_csv(ART/"model_results.csv",index=False)
json.dump({"best_model":best,"dataset_shape":list(df.shape)},open(ART/"metadata.json","w"),indent=2)
print(out.to_string(index=False))
print("Saved",ART/"best_model.pkl")
