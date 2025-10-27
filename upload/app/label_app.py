# app/label_app.py
# Radiograph QC — Labeling + Exposure (Single dataset) with corrected EI/DI

import os, sys
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# make parent dir importable so "from utils..." works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import pydicom
    HAVE_DICOM = True
except Exception:
    HAVE_DICOM = False

from utils.features import extract_features_from_image
from utils.ei_di_features import compute_ei_proxy, compute_di_proxy, ocr_ei_di_from_image

IMAGE_DIR = os.path.join("data", "images")
DATASET_CSV = os.path.join("data", "dataset.csv")
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".dcm")

CRITERIA = [
    ("positioning", "Positioning (symmetry, limbs not obscuring thorax)"),
    ("exposure",    "Exposure (vessel detail, not under/overexposed)"),
    ("collimation", "Collimation (full thorax; no large empty borders)"),
    ("sharpness",   "Sharpness/Motion (crisp borders; no blur)"),
]

# ---------- helpers ----------
def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder): return []
    return [f for f in sorted(os.listdir(folder)) if f.lower().endswith(SUPPORTED_EXTS)]

def dicom_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if np.ptp(arr) == 0: return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return (arr * 255.0).clip(0, 255).astype(np.uint8)

def read_image_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        if not HAVE_DICOM: raise RuntimeError("pydicom not installed")
        ds = pydicom.dcmread(path)
        img = ds.pixel_array
        img_u8 = dicom_to_uint8(img)
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            img_u8 = 255 - img_u8
        return img_u8
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise RuntimeError(f"Cannot read: {path}")
    return img

def load_dataset() -> pd.DataFrame:
    if os.path.exists(DATASET_CSV):
        try:
            return pd.read_csv(DATASET_CSV)
        except Exception:
            pass
    cols = ["image_name","positioning","exposure","collimation","sharpness","label",
            "laplacian_var","hist_mean","hist_std","pct_pixels_near_0","pct_pixels_near_255",
            "entropy","edge_density","symmetry_score",
            "EI_proxy","EIT_in_use","DI_proxy","EI_ocr","DI_ocr"]
    return pd.DataFrame(columns=cols)

def save_dataset(df: pd.DataFrame):
    os.makedirs(os.path.dirname(DATASET_CSV), exist_ok=True)
    df.to_csv(DATASET_CSV, index=False)

def compute_eit(images: List[str], df: pd.DataFrame, use_good_only: bool) -> float:
    candidates = images
    if use_good_only and not df.empty:
        good_names = set(df.loc[df["label"] == 1, "image_name"].dropna().tolist())
        if good_names:
            candidates = [nm for nm in images if nm in good_names]

    vals = []
    for nm in candidates:
        row = df.loc[df["image_name"] == nm]
        if not row.empty and pd.notna(row.iloc[0].get("EI_proxy")):
            vals.append(float(row.iloc[0]["EI_proxy"]))
    if not vals:
        for nm in images:
            try:
                img = read_image_any(os.path.join(IMAGE_DIR, nm))
                vals.append(compute_ei_proxy(img))
            except Exception:
                pass
    return float(np.median(vals)) if vals else 120.0

def exposure_suggestion(di_val: float) -> str:
    if di_val >= 1.0:      return "Overexposed (~+25% or more). Try ↓mAs ~10–25%."
    if 0.5 <= di_val < 1.0:return "Slightly overexposed. Consider ↓mAs ~10–15%."
    if -0.5 < di_val < 0.5:return "On target. No change suggested."
    if -1.0 < di_val <= -0.5:return "Slightly underexposed. Consider ↑mAs ~10–15%."
    return "Underexposed (~−20% or more). Try ↑mAs ~10–25%."

# ---------- chart helpers (EI/DI fixed) ----------
def _di_to_ei(eit: float, di: float) -> float:
    return float(eit * (10.0 ** (di / 10.0)))

def _hist_and_cdf(img_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist, edges = np.histogram(img_uint8.ravel(), bins=256, range=(0, 255), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cdf = np.cumsum(hist); cdf = cdf / (cdf[-1] if cdf[-1] != 0 else 1.0)
    return centers, hist, cdf

def _ei_to_intensity(ei_val: float) -> float:
    return 255.0 - float(ei_val)

def make_exposure_curves_figure(img_uint8: np.ndarray, EI_p: float, EIT: float):
    centers, hist, cdf = _hist_and_cdf(img_uint8)

    ei_lo_slight  = _di_to_ei(EIT, -0.75)
    ei_lo_clear   = _di_to_ei(EIT, -1.25)
    ei_hi_slight  = _di_to_ei(EIT,  0.75)
    ei_hi_clear   = _di_to_ei(EIT,  1.25)

    x_EIT       = _ei_to_intensity(EIT)
    x_EI_p      = _ei_to_intensity(EI_p)
    x_lo_slight = _ei_to_intensity(ei_lo_slight)
    x_lo_clear  = _ei_to_intensity(ei_lo_clear)
    x_hi_slight = _ei_to_intensity(ei_hi_slight)
    x_hi_clear  = _ei_to_intensity(ei_hi_clear)

    fig, ax1 = plt.subplots(figsize=(7, 3.2))
    ax1.plot(centers, hist, lw=1.8)
    ax1.set_xlabel("Pixel intensity (0–255)")
    ax1.set_ylabel("Density")
    ax2 = ax1.twinx()
    ax2.plot(centers, cdf, lw=1.2, linestyle="--")
    ax2.set_ylabel("CDF")

    def vline(x, color, label):
        x = max(0, min(255, x))
        ax1.axvline(x, linestyle="-", linewidth=1.2, color=color, alpha=0.9)
        ax1.text(x, ax1.get_ylim()[1]*0.92, label, rotation=90, va="top", ha="center", color=color, fontsize=9)

    vline(x_EIT,       "#444444", "EIT")
    vline(x_EI_p,      "#1f77b4", "EI_proxy")
    vline(x_lo_slight, "#d97706", "DI −0.75")
    vline(x_hi_slight, "#d97706", "DI +0.75")
    vline(x_lo_clear,  "#dc2626", "DI −1.25")
    vline(x_hi_clear,  "#dc2626", "DI +1.25")

    fig.tight_layout()
    return fig

def make_log_hist_figure(img_uint8: np.ndarray):
    centers, hist, _ = _hist_and_cdf(img_uint8)
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(centers, hist, lw=1.8)
    ax.set_xlabel("Pixel intensity (0–255)")
    ax.set_ylabel("Density (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(hist[hist>0].min()*0.5, 1e-6))
    fig.tight_layout()
    return fig

def make_regional_exposure_figure(img_uint8: np.ndarray):
    h, w = img_uint8.shape[:2]
    x1, x2 = int(w*0.2), int(w*0.8)
    y1, y2 = int(h*0.2), int(h*0.8)
    center = img_uint8[y1:y2, x1:x2]
    mask_center = np.zeros_like(img_uint8, dtype=bool)
    mask_center[y1:y2, x1:x2] = True
    border = img_uint8[~mask_center]
    hist_c, edges = np.histogram(center.ravel(), bins=256, range=(0,255), density=True)
    hist_b, _     = np.histogram(border.ravel(), bins=256, range=(0,255), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(centers, hist_c, lw=1.6, label="Center (proxy lungs)")
    ax.plot(centers, hist_b, lw=1.0, linestyle="--", label="Borders")
    ax.set_xlabel("Pixel intensity (0–255)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig

# ---------- UI ----------
st.set_page_config(page_title="Radiograph QC — Single Dataset", layout="wide")
st.title("🩻 Radiograph QC — Labeling + Exposure (Single dataset)")
st.caption("Saves to data/dataset.csv. EI/DI here are *proxies* for PNGs.")

images = list_images(IMAGE_DIR)
if not images:
    st.error(f"No images in `{IMAGE_DIR}`"); st.stop()

df = load_dataset()

# Sidebar
st.sidebar.header("Navigation & Target")
use_good_for_eit = st.sidebar.checkbox("Use Good-labeled median as EIT", value=True)
eit = compute_eit(images, df, use_good_for_eit)
eit = st.sidebar.number_input("EIT in use (proxy)", value=float(eit), step=1.0, format="%.1f")

if "idx" not in st.session_state:
    unlabeled_idx = [i for i, nm in enumerate(images)
                     if df.loc[df["image_name"] == nm].empty
                     or pd.isna(df.loc[df["image_name"] == nm].iloc[0].get("label"))]
    st.session_state.idx = unlabeled_idx[0] if unlabeled_idx else 0

idx = st.sidebar.number_input("Image index", 0, len(images)-1, st.session_state.idx, 1)
st.session_state.idx = int(idx)

colb1, colb2, colb3 = st.sidebar.columns(3)
if colb1.button("◀ Prev", use_container_width=True):
    st.session_state.idx = max(0, st.session_state.idx-1)
if colb2.button("Next ▶", use_container_width=True):
    st.session_state.idx = min(len(images)-1, st.session_state.idx+1)
if colb3.button("Next Unlabeled", use_container_width=True):
    unlabeled_idx = [i for i, nm in enumerate(images)
                     if df.loc[df["image_name"] == nm].empty
                     or pd.isna(df.loc[df["image_name"] == nm].iloc[0].get("label"))]
    if unlabeled_idx: st.session_state.idx = unlabeled_idx[0]
    else: st.toast("All images labeled ✅")

img_name = images[st.session_state.idx]
img_path = os.path.join(IMAGE_DIR, img_name)

try:
    img = read_image_any(img_path)
    disp = Image.fromarray(img)
except Exception as e:
    st.error(f"Error: {e}"); st.stop()

left, right = st.columns([2,1], gap="large")

with left:
    st.subheader(f"Image: {img_name}")
    st.image(disp, use_column_width=True, caption=img_name)

    qc = extract_features_from_image(img)
    EI_p = compute_ei_proxy(img)
    DI_p = compute_di_proxy(EI_p, eit)
    ocr = ocr_ei_di_from_image(img)

    m1, m2, m3 = st.columns(3)
    m1.metric("EI_proxy", f"{EI_p:.1f}")
    m2.metric("EIT (in use)", f"{eit:.1f}")
    m3.metric("DI_proxy", f"{DI_p:+.2f}")
    if ocr:
        st.caption(
            "Detected overlays → "
            + ("EI="+str(ocr.get("EI_ocr")) if "EI_ocr" in ocr else "")
            + ("  DI="+str(ocr.get("DI_ocr")) if "DI_ocr" in ocr else "")
        )

    sugg = exposure_suggestion(DI_p)
    color = "#16a34a" if abs(DI_p) < 0.5 else ("#d97706" if abs(DI_p) < 1.0 else "#dc2626")
    st.markdown(
        f"<div style='padding:10px;border-radius:8px;border:1px solid {color}'>"
        f"<b>Exposure suggestion:</b> <span style='color:{color}'>{sugg}</span>"
        f"</div>", unsafe_allow_html=True
    )

    with st.expander("📊 Exposure curves (Histogram + CDF)", expanded=True):
        fig1 = make_exposure_curves_figure(img, EI_p, eit)
        st.pyplot(fig1, use_container_width=True)
        st.caption("Histogram (solid) + CDF (dashed). Lines: EIT, EI_proxy, DI±0.75/±1.25 mapped to intensity.")

    with st.expander("📈 Log-density histogram (tail visibility)", expanded=False):
        fig2 = make_log_hist_figure(img)
        st.pyplot(fig2, use_container_width=True)

    with st.expander("🗺️ Regional comparison (center vs borders)", expanded=False):
        fig3 = make_regional_exposure_figure(img)
        st.pyplot(fig3, use_container_width=True)

with right:
    st.subheader("Label this image")
    row = df.loc[df["image_name"] == img_name]
    defaults = dict(positioning=1, exposure=1, collimation=1, sharpness=1, label=np.nan)
    if not row.empty:
        for k in defaults:
            v = row.iloc[0].get(k)
            if pd.notna(v):
                try: defaults[k] = int(v)
                except Exception: pass

    crit = {}
    for key, label in CRITERIA:
        idx_default = 0 if defaults.get(key,1)==1 else 1
        crit[key] = st.radio(
            label, options=[1,0],
            format_func=lambda x: "Good (1)" if x==1 else "Poor (0)",
            index=idx_default, horizontal=True, key=f"radio_{key}"
        )

    score = sum(int(crit[k]) for k,_ in CRITERIA)
    auto_label = 1 if score >= 3 else 0
    st.markdown("---")
    st.metric("Suggested Final Label (3-of-4)", "Good (1)" if auto_label==1 else "Poor (0)")

    final_label = st.radio(
        "Final Label",
        options=[1, 0],
        index=(
            0
            if not row.empty and pd.notna(row.iloc[0].get("label")) and int(row.iloc[0]["label"]) == 1
            else (0 if auto_label == 1 else 1)
        ),
        format_func=lambda x: "Good (1)" if x == 1 else "Poor (0)",
        horizontal=True
    )

    if st.button("💾 Save row", type="primary", use_container_width=True):
        new_row = {
            "image_name": img_name,
            "positioning": int(crit["positioning"]),
            "exposure": int(crit["exposure"]),
            "collimation": int(crit["collimation"]),
            "sharpness": int(crit["sharpness"]),
            "label": int(final_label),
            "laplacian_var": float(qc["laplacian_var"]),
            "hist_mean": float(qc["hist_mean"]),
            "hist_std": float(qc["hist_std"]),
            "pct_pixels_near_0": float(qc["pct_pixels_near_0"]),
            "pct_pixels_near_255": float(qc["pct_pixels_near_255"]),
            "entropy": float(qc["entropy"]),
            "edge_density": float(qc["edge_density"]),
            "symmetry_score": float(qc["symmetry_score"]),
            "EI_proxy": float(EI_p),
            "EIT_in_use": float(eit),
            "DI_proxy": float(DI_p),
            "EI_ocr": float(ocr["EI_ocr"]) if "EI_ocr" in ocr else np.nan,
            "DI_ocr": float(ocr["DI_ocr"]) if "DI_ocr" in ocr else np.nan,
        }
        df = df[df["image_name"] != img_name]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_dataset(df)
        st.success(f"Saved → data/dataset.csv · {img_name} ({'Good' if int(final_label)==1 else 'Poor'})")
        st.balloons()

    if st.button("Clear row for this image", use_container_width=True):
        df = df[df["image_name"] != img_name]
        save_dataset(df)
        st.toast("Row cleared")

st.markdown("---")
st.caption("EI/DI here are *proxies* for PNGs. For true EI/DI, use DICOM tags or console exports.")