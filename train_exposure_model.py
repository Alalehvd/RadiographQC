# train_exposure_advanced.py
# Advanced exposure model:
# - engineered features (+ optional recompute from images)
# - hyperparameter search (XGB or HistGB fallback)
# - CV threshold tuning + calibration
# - saves tuned threshold in bundle

import os, sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_fscore_support,
    roc_curve
)
from sklearn.calibration import CalibratedClassifierCV

# ---- config
DATA = "data/dataset.csv"
IMAGE_DIR = "data/images"           # where PNG/JPG/DICOM live
OUT = "models/exposure_bundle.joblib"
REPORT = "models/exposure_training_summary.json"
RANDOM_STATE = 42
N_SPLITS = 5
USE_XGB = True      # set False to force sklearn HistGradientBoosting
RECOMPUTE_FROM_IMAGES = False  # True => recompute enhanced features from image files

# macOS OpenMP (needed for xgboost on M-series)
if USE_XGB:
    os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH","")

# try to import XGBoost, otherwise fallback later
XGB_OK = False
if USE_XGB:
    try:
        from xgboost import XGBClassifier      # type: ignore
        XGB_OK = True
    except Exception:
        XGB_OK = False

if not XGB_OK:
    from sklearn.ensemble import HistGradientBoostingClassifier
    class XGBClassifier(HistGradientBoostingClassifier):  # shim
        def __init__(self, **kw):
            kw.setdefault("max_depth", 6)
            kw.setdefault("learning_rate", 0.06)
            kw.setdefault("max_iter", 600)
            kw.setdefault("random_state", RANDOM_STATE)
            kw.setdefault("validation_fraction", None)
            super().__init__(**kw)

# --- optional: use your utils to recompute enhanced features from images
sys.path.insert(0, os.path.abspath("."))  # allow "from utils..." from project root
try:
    from utils.features import extract_features_from_image
    from utils.ei_di_features import compute_ei_proxy, compute_di_proxy
    HAVE_UTILS = True
except Exception:
    HAVE_UTILS = False

# -------- Feature engineering helpers --------
def safe_pct(a: np.ndarray, thresh: int, side: str) -> float:
    n = a.size if a.size else 1
    if side == "low":
        return float((a <= thresh).sum() / n)
    else:
        return float((a >= thresh).sum() / n)

def local_contrast(img: np.ndarray, k: int = 7) -> float:
    # quick local std as contrast proxy
    import cv2
    imgf = img.astype(np.float32)
    mean = cv2.boxFilter(imgf, -1, (k, k), normalize=True)
    sq = cv2.boxFilter(imgf * imgf, -1, (k, k), normalize=True)
    var = np.maximum(sq - mean * mean, 0.0)
    return float(np.mean(np.sqrt(var + 1e-6)))

def regional_stats(img: np.ndarray) -> Dict[str, float]:
    h, w = img.shape[:2]
    x1, x2 = int(w*0.2), int(w*0.8)
    y1, y2 = int(h*0.2), int(h*0.8)
    center = img[y1:y2, x1:x2].ravel().astype(np.float32)
    border = img.copy().ravel().astype(np.float32)
    mask = np.zeros((h,w), dtype=bool); mask[y1:y2, x1:x2] = True
    border = img[~mask].ravel().astype(np.float32)

    def stats(v: np.ndarray, prefix: str) -> Dict[str, float]:
        if v.size == 0:
            return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
                    f"{prefix}_p10": np.nan, f"{prefix}_p50": np.nan, f"{prefix}_p90": np.nan}
        return {
            f"{prefix}_mean": float(np.mean(v)),
            f"{prefix}_std": float(np.std(v)),
            f"{prefix}_p10": float(np.percentile(v, 10)),
            f"{prefix}_p50": float(np.percentile(v, 50)),
            f"{prefix}_p90": float(np.percentile(v, 90)),
        }

    feats = {}
    feats.update(stats(center, "ctr"))
    feats.update(stats(border, "brd"))
    feats["ctr_brd_mean_diff"] = feats["ctr_mean"] - feats["brd_mean"]
    feats["ctr_brd_std_ratio"] = (feats["ctr_std"] / (feats["brd_std"] + 1e-6))
    return feats

def engineer_from_image(img: np.ndarray, eit_proxy: float) -> Dict[str, float]:
    """Lightweight engineered features from raw uint8 image."""
    arr = img.astype(np.uint8).ravel()
    p01, p99 = np.percentile(arr, [1, 99])
    p10, p50, p90 = np.percentile(arr, [10, 50, 90])
    clip_low = safe_pct(arr, 3, "low")
    clip_hi  = safe_pct(arr, 252, "high")

    feats = {
        "px_mean": float(arr.mean()),
        "px_std": float(arr.std(ddof=0)),
        "px_p01": float(p01), "px_p10": float(p10), "px_p50": float(p50),
        "px_p90": float(p90), "px_p99": float(p99),
        "clip_low_pct": float(clip_low),
        "clip_hi_pct": float(clip_hi),
        "local_contrast": local_contrast(img, 7),
    }
    feats.update(regional_stats(img))
    return feats

# ---------- Model utils ----------
def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])

@dataclass
class CVResult:
    aucs: List[float]
    f1s: List[float]
    thrs: List[float]

def cross_val_with_threshold(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, opt: str = "f1") -> CVResult:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    aucs, f1s, thrs = [], [], []
    for tr, te in skf.split(X, y):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        # calibrate on train for better probs
        Xp_tr = pipe["prep"].transform(X.iloc[tr])
        cal = CalibratedClassifierCV(pipe["clf"], method="sigmoid", cv=3)
        cal.fit(Xp_tr, y.iloc[tr])

        Xp_te = pipe["prep"].transform(X.iloc[te])
        prob = cal.predict_proba(Xp_te)[:,1]

        if opt == "f1":
            # search best threshold on validation set
            th_candidates = np.linspace(0.1, 0.9, 33)
            f1_best, thr_best = -1, 0.5
            for th in th_candidates:
                pred = (prob >= th).astype(int)
                f1 = f1_score(y.iloc[te], pred, zero_division=0)
                if f1 > f1_best:
                    f1_best, thr_best = f1, th
            th = thr_best
        else:
            th = youden_threshold(y.iloc[te].values, prob)

        pred = (prob >= th).astype(int)
        aucs.append(roc_auc_score(y.iloc[te], prob))
        f1s.append(f1_score(y.iloc[te], pred, zero_division=0))
        thrs.append(float(th))
    return CVResult(aucs, f1s, thrs)

def build_model_and_search(num_cols: List[str], y: pd.Series):
    pre = ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="drop")

    if XGB_OK:
        base = XGBClassifier(
            n_estimators=700, max_depth=5, learning_rate=0.055,
            subsample=0.9, colsample_bytree=0.8, reg_lambda=1.2,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
            scale_pos_weight=( (len(y)-y.sum()) / max(1,y.sum()) )
        )
        param_dist = {
            "clf__n_estimators": [500, 700, 900],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.03, 0.05, 0.07],
            "clf__subsample": [0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__reg_lambda": [0.8, 1.0, 1.5, 2.0],
        }
    else:
        base = XGBClassifier()  # HistGB shim
        param_dist = {
            "clf__max_depth": [4, 6, 8],
            "clf__learning_rate": [0.04, 0.06, 0.08],
            "clf__max_iter": [400, 600, 800],
        }

    pipe = Pipeline([("prep", pre), ("clf", base)])
    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=20,
        scoring="roc_auc", cv=N_SPLITS, random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    return search

# ---------- Main ----------
def main():
    if not os.path.exists(DATA):
        raise SystemExit(f"Missing {DATA}. Label in app first.")

    df = pd.read_csv(DATA)
    if "exposure" not in df.columns:
        raise SystemExit("dataset.csv must include 'exposure' (criterion 1/0).")

    # Base feature set present in dataset
    base_feature_cols = [
        # existing numeric features we expect in dataset.csv
        "laplacian_var","hist_mean","hist_std","pct_pixels_near_0","pct_pixels_near_255",
        "entropy","edge_density","symmetry_score","EI_proxy","EIT_in_use","DI_proxy",
        "EI_ocr","DI_ocr"
    ]
    for c in base_feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Optionally recompute + extend features from raw images
    extra_cols = []
    if RECOMPUTE_FROM_IMAGES and HAVE_UTILS:
        recs = []
        for _, row in df.iterrows():
            name = row.get("image_name")
            if not isinstance(name, str): 
                recs.append({})
                continue
            # try reading PNG/JPG first; DICOM optional
            img_path_png = Path(IMAGE_DIR) / name
            if not img_path_png.exists():
                # allow name without ext? (skip for now)
                recs.append({})
                continue
            # read as grayscale
            import cv2
            img = cv2.imread(str(img_path_png), cv2.IMREAD_GRAYSCALE)
            if img is None:
                recs.append({})
                continue
            feats = engineer_from_image(img, row.get("EIT_in_use", np.nan))
            recs.append(feats)
        extra_df = pd.DataFrame(recs)
        extra_cols = list(extra_df.columns)
        df = pd.concat([df, extra_df], axis=1)

    # Build training table
    use_cols = [c for c in (base_feature_cols + extra_cols) if c in df.columns]
    y = df["exposure"].astype(int)
    X = df[use_cols].copy()

    # Make numeric, fillna
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    # Hyperparam search (AUC)
    search = build_model_and_search(use_cols, y)
    search.fit(X, y)
    best_pipe = search.best_estimator_
    best_params = search.best_params_
    best_auc = search.best_score_
    print(f"\nBest AUC (CV): {best_auc:.3f}")
    print("Best params:", best_params)

    # CV with threshold tuning (F1)
    cvres = cross_val_with_threshold(best_pipe, X, y, opt="f1")
    tuned_thr = float(np.median(cvres.thrs))
    print(f"Tuned threshold (median across folds): {tuned_thr:.3f}")
    print(f"CV AUC mean±sd: {np.mean(cvres.aucs):.3f}±{np.std(cvres.aucs):.3f}")
    print(f"CV F1  mean±sd: {np.mean(cvres.f1s):.3f}±{np.std(cvres.f1s):.3f}")

    # Fit on full data + calibrate
    best_pipe.fit(X, y)
    Xp = best_pipe["prep"].transform(X)
    cal = CalibratedClassifierCV(best_pipe["clf"], method="sigmoid", cv=3)
    cal.fit(Xp, y)

    bundle = {
        "prep": best_pipe["prep"],
        "clf": best_pipe["clf"],
        "cal": cal,
        "features": use_cols,
        "threshold": tuned_thr,
        "meta": {
            "model": "xgboost" if XGB_OK else "histgb",
            "random_state": RANDOM_STATE,
            "n_splits": N_SPLITS,
            "best_auc_cv": float(best_auc),
            "cv_auc_mean": float(np.mean(cvres.aucs)),
            "cv_auc_sd": float(np.std(cvres.aucs)),
            "cv_f1_mean": float(np.mean(cvres.f1s)),
            "cv_f1_sd": float(np.std(cvres.f1s)),
            "recomputed_from_images": bool(RECOMPUTE_FROM_IMAGES),
            "extra_cols": extra_cols,
        }
    }
    os.makedirs(Path(OUT).parent, exist_ok=True)
    joblib.dump(bundle, OUT)
    print(f"✅ Saved exposure bundle → {OUT}")

    with open(REPORT, "w") as f:
        json.dump(bundle["meta"], f, indent=2)
    print(f"📄 Summary → {REPORT}")

if __name__ == "__main__":
    main()