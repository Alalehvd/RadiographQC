# scripts/enrich_dataset_exposure_features.py
# Adds dark-tail and percentile features to data/dataset.csv (keeps labels intact)

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT/ "radiograph_qc" / "data" / "dataset.csv"
IMG_DIR  = ROOT/ "radiograph_qc"  / "data" / "images"

sys.path.insert(0, str(ROOT))
from utils.ei_di_features import exposure_aux_features, compute_ei_proxy, compute_di_proxy

import cv2

def read_gray(path: Path):
    if path.suffix.lower() == ".dcm":
        import pydicom
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        ptp = np.ptp(arr)
        if ptp == 0: arr = np.zeros_like(arr)
        else:        arr = (arr - arr.min()) / ptp
        img = (arr * 255.0).astype(np.uint8)
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            img = 255 - img
        return img
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img

def main():
    if not DATA_CSV.exists():
        raise SystemExit(f"Missing {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)

    add_cols = ["tail_le_10","tail_le_20","tail_le_30","clip_le_3","p10","p25","p50","p75","p90"]
    for c in add_cols:
        if c not in df.columns: df[c] = np.nan

    # ensure EI/DI columns exist too
    for c in ["EI_proxy","EIT_in_use","DI_proxy"]:
        if c not in df.columns: df[c] = np.nan

    # if EIT_in_use is NaN, use median EI of Good images later
    for i, row in df.iterrows():
        name = str(row["image_name"])
        path = IMG_DIR / name
        if not path.exists(): 
            continue
        img = read_gray(path)
        aux = exposure_aux_features(img)
        for k, v in aux.items():
            df.at[i, k] = float(v)

        # (Re)compute EI if missing
        if pd.isna(row.get("EI_proxy")):
            df.at[i, "EI_proxy"] = float(compute_ei_proxy(img))

    # Compute EIT if missing: median EI of label==1
    if df["EIT_in_use"].isna().all():
        good = pd.to_numeric(df.loc[df["label"] == 1, "EI_proxy"], errors="coerce").dropna()
        df["EIT_in_use"] = float(np.median(good)) if len(good) else 120.0

    # Recompute DI from EI/EIT
    def _di(r):
        try:
            return float(compute_di_proxy(float(r["EI_proxy"]), float(r["EIT_in_use"])))
        except Exception:
            return np.nan
    df["DI_proxy"] = df.apply(_di, axis=1)

    df.to_csv(DATA_CSV, index=False)
    print(f"✅ Enriched and saved: {DATA_CSV}")

if __name__ == "__main__":
    main()