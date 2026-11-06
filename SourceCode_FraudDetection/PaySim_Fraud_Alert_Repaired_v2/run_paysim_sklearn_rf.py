"""scikit-learn Random Forest counterpart to run_paysim_spark.py.

Uses the same pre-event PaySim feature definitions and step-based split as the
Spark MLlib script, allowing a more comparable library experiment. This full
6.08M-row RF run may take several minutes and substantial RAM.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
from src.paysim_features import DATA_PATH, load_paysim, build_features, BALANCE_FEATURES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "sklearn_random_forest"


def evaluate(y, scores, threshold):
    y = np.asarray(y, dtype=np.int8); scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0,1]).ravel()]
    return {"rows": int(len(y)), "fraud_actual": int(y.sum()), "nonfraud_actual": int(len(y)-y.sum()),
            "threshold": float(threshold), "alerts": int(pred.sum()), "true_positive": tp,
            "false_positive": fp, "true_negative": tn, "false_negative": fn,
            "precision": float(precision_score(y,pred,zero_division=0)),
            "recall": float(recall_score(y,pred,zero_division=0)), "f1": float(f1_score(y,pred,zero_division=0)),
            "accuracy": float(accuracy_score(y,pred)), "roc_auc": float(roc_auc_score(y,scores)),
            "average_precision": float(average_precision_score(y,scores))}


def budget(y, scores):
    order=np.argsort(-scores,kind="mergesort"); out=[]
    for pct in [0.005,0.01,0.03,0.05,0.10,0.20,0.30,0.50]:
        k=max(1,int(round(len(y)*pct))); tp=int(np.asarray(y)[order[:k]].sum()); fp=k-tp
        out.append({"alert_budget_percent":100*pct,"alerts":k,"true_positive":tp,"false_positive":fp,
                    "false_negative":int(np.asarray(y).sum())-tp,"precision":tp/k,"recall":tp/int(np.asarray(y).sum())})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_paysim(DATA_PATH, usecols=["step","type","amount","oldbalanceOrg","oldbalanceDest","isFraud"])
    step = raw["step"].astype(int)
    m_train = step <= 520; m_val=(step>520)&(step<=600); m_test=step>600
    y_train=raw.loc[m_train,"isFraud"].to_numpy(dtype=np.int8)
    y_val=raw.loc[m_val,"isFraud"].to_numpy(dtype=np.int8)
    y_test=raw.loc[m_test,"isFraud"].to_numpy(dtype=np.int8)
    split=[]
    for label,mask in [("train",m_train),("validation",m_val),("test",m_test)]:
        part=raw.loc[mask]
        split.append({"partition":label,"rows":len(part),"fraud_count":int(part.isFraud.sum()),"step_min":int(part.step.min()),"step_max":int(part.step.max())})
    pd.DataFrame(split).to_csv(OUT/"temporal_split_summary.csv",index=False)

    X = build_features(raw, include_balance_features=True).loc[:, BALANCE_FEATURES]
    num_cols=[c for c in BALANCE_FEATURES if c!="type"]
    prep=ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["type"]),
    ], remainder="drop")
    rf=RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=20,
                              max_features="sqrt", max_samples=0.8,
                              random_state=42, n_jobs=-1, class_weight=None)
    pipe=Pipeline([("preprocess",prep),("model",rf)])
    print(f"Training scikit-learn RandomForest on {int(m_train.sum()):,} rows; this may take several minutes.",flush=True)
    start=time.perf_counter(); pipe.fit(X.loc[m_train],y_train); fit_seconds=time.perf_counter()-start
    val_scores=pipe.predict_proba(X.loc[m_val])[:,1]
    p,r,t=precision_recall_curve(y_val,val_scores)
    f=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros_like(p[:-1]),where=(p[:-1]+r[:-1])>0)
    threshold=float(t[int(np.nanargmax(f))]) if len(t) else 0.5
    validation_metrics=evaluate(y_val,val_scores,threshold)
    test_scores=pipe.predict_proba(X.loc[m_test])[:,1]
    test_metrics=evaluate(y_test,test_scores,threshold)
    default_metrics=evaluate(y_test,test_scores,0.5)
    budgets=budget(y_test,test_scores)
    metrics={"implementation":"scikit-learn RandomForestClassifier","dataset":DATA_PATH.name,
             "model_config":{"n_estimators":80,"max_depth":10,"min_samples_leaf":20,"max_features":"sqrt","max_samples":0.8},
             "split":split,"threshold_selection":"maximum F1 on validation only","threshold":threshold,
             "validation_metrics":validation_metrics,"test_metrics_at_validation_threshold":test_metrics,
             "test_metrics_at_default_threshold_0_5":default_metrics,"test_alert_budgets":budgets,
             "training_seconds":fit_seconds,
             "excluded_features":["isFraud-as-input","isFlaggedFraud","newbalanceOrig","newbalanceDest","nameOrig","nameDest"],
             "warning":"PaySim is synthetic. Compare with Spark using identical split/features, but local runtime is not a cluster scalability test."}
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    pd.DataFrame([test_metrics]).to_csv(OUT/"test_metrics.csv",index=False)
    pd.DataFrame(budgets).to_csv(OUT/"test_alert_budget_metrics.csv",index=False)
    tn,fp,fn,tp=[int(v) for v in confusion_matrix(y_test,(test_scores>=threshold).astype(int),labels=[0,1]).ravel()]
    pd.DataFrame([[tn,fp],[fn,tp]],index=["actual_nonfraud","actual_fraud"],columns=["pred_nonfraud","pred_fraud"]).to_csv(OUT/"confusion_matrix.csv")
    names=pipe.named_steps["preprocess"].get_feature_names_out()
    imp=pipe.named_steps["model"].feature_importances_
    pd.DataFrame({"feature":names,"importance":imp}).sort_values("importance",ascending=False).to_csv(OUT/"feature_importance.csv",index=False)
    pd.DataFrame({"step":raw.loc[m_test,"step"].to_numpy(),"type":raw.loc[m_test,"type"].astype(str).to_numpy(),
                  "amount":raw.loc[m_test,"amount"].to_numpy(),"label":y_test,"score":test_scores,
                  "prediction":(test_scores>=threshold).astype(int)}).to_csv(OUT/"test_predictions.csv",index=False)
    print("TEST metrics:",json.dumps(test_metrics,indent=2),flush=True)
    print(pd.DataFrame(budgets).to_string(index=False),flush=True)

if __name__=="__main__": main()
