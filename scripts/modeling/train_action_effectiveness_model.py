#!/usr/bin/env python3
"""Fast training script for Part6 backend demo models.

It trains lightweight sklearn pipelines and saves them using the same filenames
that your future backend can load. For final paper experiments you can replace
these estimators with full LightGBM/XGBoost runs; the feature schema stays the same.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "label_positive_excess_12m"
DATE_COL = "report_date"
META = {"event_id","training_window_years","training_window_months","manager","fund","crsp_portno","crsp_fundno","fund_ticker","mgmt_name","report_date","year","quarter","month_key","feature_cutoff_date","label_start_date","label_end_date"}
LABELS = {"label_positive_excess_12m","label_positive_excess_4q","label_downside_control_12m","label_joint_good_12m","future_12m_excess_return","future_drawdown"}


def infer_numeric(df):
    cols=[]
    for c in df.columns:
        if c in META or c in LABELS:
            continue
        s=pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 50:
            cols.append(c)
    return cols


def time_split(df):
    df=df.copy()
    df[DATE_COL]=pd.to_datetime(df[DATE_COL], errors="coerce")
    df=df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    tr=df[df[DATE_COL] < "2017-01-01"]
    va=df[(df[DATE_COL] >= "2017-01-01") & (df[DATE_COL] < "2021-01-01")]
    te=df[df[DATE_COL] >= "2021-01-01"]
    if min(len(tr),len(va),len(te)) < 100:
        n=len(df); tr=df.iloc[:int(.7*n)]; va=df.iloc[int(.7*n):int(.85*n)]; te=df.iloc[int(.85*n):]
    return tr,va,te


def metrics(model, frame, features):
    y=frame[TARGET].astype(int)
    X=frame[features]
    p=model.predict_proba(X)[:,1]
    pred=(p>=.5).astype(int)
    def auc(fn):
        try:
            return float(fn(y,p)) if len(set(y))>1 else None
        except Exception:
            return None
    return {"rows":int(len(frame)),"positive_rate":float(y.mean()),"auc":auc(roc_auc_score),"average_precision":auc(average_precision_score),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"f1":float(f1_score(y,pred,zero_division=0))}


def train_dataset(path:Path, model_dir:Path, max_rows:int, seed:int):
    df=pd.read_csv(path, low_memory=False)
    df=df[pd.to_numeric(df[TARGET], errors="coerce").notna()].copy()
    df[TARGET]=pd.to_numeric(df[TARGET], errors="coerce").astype(int)
    if len(df)>max_rows:
        # balanced sample for quick local artifact generation
        parts=[]
        for _,g in df.groupby(TARGET):
            parts.append(g.sample(min(len(g), max_rows//2), random_state=seed))
        df=pd.concat(parts).sort_values(DATE_COL).reset_index(drop=True)
    features=infer_numeric(df)
    horizon=int(df["training_window_years"].dropna().iloc[0]) if "training_window_years" in df else 0
    tr,va,te=time_split(df)
    estimators={
        "lightgbm": LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed),
        "xgboost": RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=seed, n_jobs=-1),
    }
    result={"dataset":str(path),"horizon_years":horizon,"target":TARGET,"numeric_features":features,"splits":{"train":len(tr),"valid":len(va),"test":len(te)},"models":{}}
    for name,est in estimators.items():
        steps=[("imputer",SimpleImputer(strategy="median"))]
        if name=="lightgbm": steps.append(("scaler",StandardScaler()))
        steps.append(("model",est))
        model=Pipeline(steps)
        model.fit(tr[features], tr[TARGET].astype(int))
        out=model_dir/f"{name}_action_model_trailing{horizon}y.pkl"
        joblib.dump(model,out)
        result["models"][name]={"path":str(out),"metrics":{"train":metrics(model,tr,features),"valid":metrics(model,va,features),"test":metrics(model,te,features)}}
    bg=tr.sample(min(len(tr),500), random_state=seed)[features]
    bg_csv=model_dir/f"shap_background_sample_trailing{horizon}y.csv"
    bg.to_csv(bg_csv,index=False,encoding="utf-8-sig")
    result["shap_background_sample"]=str(bg_csv)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,default=Path("data"))
    ap.add_argument("--model-dir",type=Path,default=None)
    ap.add_argument("--max-rows",type=int,default=8000)
    ap.add_argument("--random-state",type=int,default=42)
    a=ap.parse_args()
    pred=a.data_root/"derived"/"prediction"
    model_dir=a.model_dir or (a.data_root.parent/"models"/"action_effectiveness"/"v001")
    model_dir.mkdir(parents=True,exist_ok=True)
    results={}
    for y in [3,5]:
        p=pred/f"part6_prediction_dataset_trailing{y}y_future12m.csv"
        if not p.exists():
            print(f"[WARN] missing {p}"); continue
        print(f"[TRAIN] {p}")
        r=train_dataset(p,model_dir,a.max_rows,a.random_state)
        results[str(y)]=r
    for name in ["lightgbm","xgboost"]:
        src=model_dir/f"{name}_action_model_trailing3y.pkl"
        dst=model_dir/f"{name}_action_model.pkl"
        if src.exists(): shutil.copyfile(src,dst)
    feature_meta={"target":TARGET,"default_horizon_years":3,"note":"Demo artifacts use fast sklearn estimators saved with backend-compatible names; replace with full LightGBM/XGBoost for final paper experiments.","horizon_specific":{k:{"numeric_features":v["numeric_features"],"categorical_features":[]} for k,v in results.items()}}
    (model_dir/"feature_columns.json").write_text(json.dumps(feature_meta,ensure_ascii=False,indent=2),encoding="utf-8")
    (model_dir/"preprocessing_config.json").write_text(json.dumps({"numeric":"median imputation; logistic model additionally standardizes numeric features","categorical":"not used in fast demo script","target":TARGET},ensure_ascii=False,indent=2),encoding="utf-8")
    (model_dir/"model_metadata.json").write_text(json.dumps({"models":results},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[DONE] {model_dir}")
if __name__=="__main__": main()
