# train_models_full.py
# Trains two models using ALL available details:
#  - exposure model (target: exposure) -> uses all numeric features except the target
#  - quality  model (target: label)    -> uses all numeric features + the 4 criteria radios
#
# Saves:
#  models/exposure_bundle.joblib
#  models/quality_bundle.joblib

import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA = "data/dataset.csv"
os.makedirs("models", exist_ok=True)

CRITERION_COLS = ["positioning","exposure","collimation","sharpness"]

def build_pipeline(numeric_cols, scale_pos_weight=1.0):
    pre = ColumnTransformer([("num", StandardScaler(), numeric_cols)], remainder="drop")
    clf = XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss"
    )
    return Pipeline([("prep", pre), ("clf", clf)])

def kfold_report(pipe, X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs, f1s = [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        proba = pipe.predict_proba(X.iloc[te])[:,1]
        pred  = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y.iloc[te], proba)
        p, r, f1, _ = precision_recall_fscore_support(y.iloc[te], pred, average="binary", zero_division=0)
        aucs.append(auc); f1s.append(f1)
        print(f"[{name}] Fold {fold}  AUC={auc:.3f}  F1={f1:.3f}")
    print(f"[{name}] Mean AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")

def train_and_save(task_name, df, target_col, drop_cols):
    y = df[target_col].astype(int)
    X = df.drop(columns=drop_cols, errors="ignore")
    # keep only numeric columns (criteria are ints; OCR/metrics are floats)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(df[c])]
    X = X[num_cols]

    pos = y.sum(); neg = len(y) - pos
    spw = (neg / max(pos,1))
    pipe = build_pipeline(num_cols, scale_pos_weight=spw)

    print(f"\n=== {task_name.upper()} ===")
    print(f"Samples: {len(y)} | Positives: {int(pos)} | Negatives: {int(neg)}")
    print(f"Features ({len(num_cols)}): {', '.join(num_cols)}")

    kfold_report(pipe, X, y, task_name)

    # Fit on full + calibrate
    pipe.fit(X, y)
    Xp = pipe["prep"].transform(X)
    cal = CalibratedClassifierCV(pipe["clf"], method="sigmoid", cv=3)
    cal.fit(Xp, y)

    out_path = f"models/{task_name}_bundle.joblib"
    joblib.dump({"prep": pipe["prep"], "clf": pipe["clf"], "cal": cal, "features": num_cols}, out_path)
    print(f"✅ Saved {task_name} → {out_path}")

def main():
    df = pd.read_csv(DATA)
    need = {"image_name","label","exposure"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{DATA} must contain: {need}")

    # EXPOSURE model: use ALL numeric features EXCEPT the target itself
    drop_exp = ["image_name","label","exposure"]  # don't leak target; drop non-features
    train_and_save("exposure", df.dropna(subset=["exposure"]), "exposure", drop_exp)

    # QUALITY model: use ALL numeric features + the 4 criteria
    # (we predict 'label', so we drop label + image_name only)
    drop_qual = ["image_name","label"]
    train_and_save("quality", df.dropna(subset=["label"]), "label", drop_qual)

if __name__ == "__main__":
    # macOS OpenMP (if needed)
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH","")
    main()