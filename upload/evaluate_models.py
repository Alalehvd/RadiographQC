# evaluate_models.py
# Hold-out evaluation for:
#   - Exposure model (target: exposure)
#   - Quality model (target: label)
#
# Outputs:
#   reports/YYYYMMDD_HHMM/
#     - exposure_*.png
#     - quality_*.png
#     - report.md

import os, sys, time, joblib, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, brier_score_loss,
    classification_report
)
from sklearn.calibration import CalibratedClassifierCV

# ===== Choose model family =====
USE_XGBOOST = True  # set False to force sklearn's HistGradientBoosting

# Optional on macOS for XGBoost OpenMP
if USE_XGBOOST:
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH","")

if USE_XGBOOST:
    try:
        from xgboost import XGBClassifier
        XGB_OK = True
    except Exception:
        XGB_OK = False
else:
    XGB_OK = False

if not XGB_OK:
    from sklearn.ensemble import HistGradientBoostingClassifier
    # make a shim with similar args for simplicity
    class XGBClassifier(HistGradientBoostingClassifier):
        def __init__(self, max_depth=6, learning_rate=0.06, max_iter=500,
                     random_state=42, class_weight=None, **kwargs):
            super().__init__(max_depth=max_depth, learning_rate=learning_rate,
                             max_iter=max_iter, validation_fraction=None,
                             random_state=random_state, class_weight=class_weight)

DATA = "data/dataset.csv"
OUTDIR = Path("reports") / datetime.now().strftime("%Y%m%d_%H%M")
OUTDIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def build_pipeline(numeric_cols, y):
    pre = ColumnTransformer([("num", StandardScaler(), numeric_cols)], remainder="drop")
    pos = int(np.sum(y)); neg = len(y) - pos
    spw = (neg / max(pos, 1))  # imbalance handle

    if XGB_OK:
        clf = XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        # scale_pos_weight only for xgboost
        clf.set_params(scale_pos_weight=spw)
    else:
        # HistGradientBoosting: use class_weight
        clf = XGBClassifier(max_depth=6, learning_rate=0.06, max_iter=500,
                            random_state=RANDOM_STATE,
                            class_weight={0:1.0, 1:float(spw)})

    pipe = Pipeline([("prep", pre), ("clf", clf)])
    return pipe

def eval_and_plots(name, df, target_col, drop_cols):
    print(f"\n=== {name.upper()} (target: {target_col}) ===")
    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].astype(int)
    X = df.drop(columns=drop_cols, errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    pipe = build_pipeline(X.columns.tolist(), y_train)
    pipe.fit(X_train, y_train)

    # Calibrate on train (internal CV)
    Xp_train = pipe["prep"].transform(X_train)
    cal = CalibratedClassifierCV(pipe["clf"], method="sigmoid", cv=3)
    cal.fit(Xp_train, y_train)

    # Predict on test
    Xp_test = pipe["prep"].transform(X_test)
    proba = cal.predict_proba(Xp_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, proba)
    ap  = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, digits=3)

    print(f"AUC: {auc:.3f} | AP: {ap:.3f} | Brier: {brier:.4f}")
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)

    # ===== Plots =====
    # ROC
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1],[0,1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name} ROC (AUC={auc:.3f})")
    plt.tight_layout()
    roc_path = OUTDIR / f"{name}_roc.png"
    plt.savefig(roc_path, dpi=160); plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.figure(figsize=(5,4))
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{name} PR (AP={ap:.3f})")
    plt.tight_layout()
    pr_path = OUTDIR / f"{name}_pr.png"
    plt.savefig(pr_path, dpi=160); plt.close()

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(4.2,4))
    disp.plot(ax=ax, colorbar=False)
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    cm_path = OUTDIR / f"{name}_cm.png"
    plt.savefig(cm_path, dpi=160); plt.close()

    # Calibration curve (reliability)
    # Bin predicted probabilities into deciles
    bins = np.linspace(0.0, 1.0, 11)
    inds = np.digitize(proba, bins) - 1
    prob_bin = []
    frac_pos = []
    for b in range(10):
        mask = inds == b
        if mask.sum() > 0:
            prob_bin.append(proba[mask].mean())
            frac_pos.append(y_test[mask].mean())
    plt.figure(figsize=(5,4))
    plt.plot([0,1],[0,1], "k--", lw=1)
    plt.plot(prob_bin, frac_pos, marker="o")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"{name} Calibration")
    plt.tight_layout()
    cal_path = OUTDIR / f"{name}_calibration.png"
    plt.savefig(cal_path, dpi=160); plt.close()

    # Feature importance (tree-based only)
    fi_path = None
    try:
        importances = None
        if hasattr(pipe["clf"], "feature_importances_"):
            importances = pipe["clf"].feature_importances_
        elif hasattr(pipe["clf"], "feature_importances"):
            importances = pipe["clf"].feature_importances_
        if importances is not None:
            imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(20)
            plt.figure(figsize=(6,5))
            imp[::-1].plot(kind="barh")
            plt.title(f"{name} Top Feature Importances")
            plt.tight_layout()
            fi_path = OUTDIR / f"{name}_feature_importance.png"
            plt.savefig(fi_path, dpi=160); plt.close()
    except Exception:
        pass

    # Save a small bundle for reproducibility (optional)
    joblib.dump({"prep": pipe["prep"], "clf": pipe["clf"], "cal": cal, "features": X.columns.tolist()},
                OUTDIR / f"{name}_eval_bundle.joblib")

    # Return summary for the markdown report
    return {
        "name": name,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "pos_rate_test": float(y_test.mean()),
        "auc": float(auc),
        "ap": float(ap),
        "brier": float(brier),
        "cm": cm.tolist(),
        "report": report,
        "paths": {
            "roc": str(roc_path),
            "pr": str(pr_path),
            "cm": str(cm_path),
            "cal": str(cal_path),
            "fi": str(fi_path) if fi_path else None
        }
    }

def main():
    if not os.path.exists(DATA):
        raise SystemExit(f"Missing {DATA}. Label first in the app.")

    df = pd.read_csv(DATA)
    # Basic sanity
    if "image_name" not in df.columns:
        raise SystemExit("dataset.csv must contain an 'image_name' column")

    # Common drops (non-features)
    drop_common = ["image_name"]

    # Evaluate Exposure model
    exp_summary = eval_and_plots(
        name="exposure",
        df=df,
        target_col="exposure",
        drop_cols=drop_common + ["label"]  # don't leak final label
    )

    # Evaluate Final Quality model
    qual_summary = eval_and_plots(
        name="quality",
        df=df,
        target_col="label",
        drop_cols=drop_common  # we allow criterion columns to be used as signals if present
    )

    # Write markdown report
    md = OUTDIR / "report.md"
    with open(md, "w") as f:
        f.write(f"# Radiograph QC — Hold-out Evaluation\n")
        f.write(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n\n")

        for s in [exp_summary, qual_summary]:
            f.write(f"## {s['name'].title()} Model\n")
            f.write(f"- Train/Test sizes: {s['n_train']} / {s['n_test']}\n")
            f.write(f"- Test positive rate: {s['pos_rate_test']:.3f}\n")
            f.write(f"- AUC: **{s['auc']:.3f}**\n")
            f.write(f"- Average Precision: **{s['ap']:.3f}**\n")
            f.write(f"- Brier score: {s['brier']:.4f}\n\n")
            f.write("### Confusion Matrix\n")
            f.write(f"![cm]({Path(s['paths']['cm']).name})\n\n")
            f.write("### ROC Curve\n")
            f.write(f"![roc]({Path(s['paths']['roc']).name})\n\n")
            f.write("### Precision–Recall Curve\n")
            f.write(f"![pr]({Path(s['paths']['pr']).name})\n\n")
            f.write("### Calibration Curve\n")
            f.write(f"![cal]({Path(s['paths']['cal']).name})\n\n")
            if s['paths']['fi']:
                f.write("### Top Feature Importances\n")
                f.write(f"![fi]({Path(s['paths']['fi']).name})\n\n")
            f.write("### Classification Report\n")
            f.write("```\n")
            f.write(s["report"])
            f.write("```\n\n")

    print(f"\n✅ Report written to: {OUTDIR / 'report.md'}")
    print("Open the PNGs in that folder to view the plots.")

if __name__ == "__main__":
    main()