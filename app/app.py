# app/predict_app.py
# Radiograph QC — Upload & Predict (sensitive exposure features + tuned threshold)

import os, sys, io
from typing import Tuple
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import joblib
import matplotlib.pyplot as plt

# Import utils path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent   # repo root (since app/ is one level down)
MODEL_DIR = ROOT / "models"

# Optional DICOM
try:
    import pydicom
    HAVE_DICOM = True
except Exception:
    HAVE_DICOM = False

from utils.features import extract_features_from_image
from utils.ei_di_features import (
    compute_ei_proxy, compute_di_proxy, ocr_ei_di_from_image, exposure_aux_features
)

DATASET_CSV = os.path.join("data", "dataset.csv")
SUPPORTED = (".png", ".jpg", ".jpeg", ".dcm")
CRITERION_COLS = {"positioning", "exposure", "collimation", "sharpness"}

# ---------- I/O ----------
def read_image_any_bytes(name: str, byts: bytes) -> np.ndarray:
    ext = os.path.splitext(name)[1].lower()
    if ext == ".dcm":
        if not HAVE_DICOM:
            raise RuntimeError("DICOM uploaded but pydicom is not installed.")
        ds = pydicom.dcmread(io.BytesIO(byts))
        arr = ds.pixel_array.astype(np.float32)
        ptp = np.ptp(arr)
        if ptp == 0: arr = np.zeros_like(arr)
        else:        arr = (arr - arr.min()) / ptp
        img_u8 = (arr * 255.0).astype(np.uint8)
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            img_u8 = 255 - img_u8
        return img_u8
    pil = Image.open(io.BytesIO(byts)).convert("L")
    return np.array(pil, dtype=np.uint8)

def _load_bundle(name: str):
    p = (MODEL_DIR / name)
    try:
        if not p.exists():
            st.sidebar.warning(f"Model file not found: {p}")
            return None
        bundle = joblib.load(p)
        return bundle
    except Exception as e:
        st.sidebar.error(f"Failed to load {name}: {type(e).__name__}: {e}")
        return None

# ---------- Features / prediction ----------
def align_to_bundle_features(bundle, feats_df: pd.DataFrame) -> pd.DataFrame:
    need = bundle.get("features", [])
    out = feats_df.copy()
    for col in need:
        if col not in out.columns:
            out[col] = 1 if col in CRITERION_COLS else 0.0
    out = out[need]
    for c in out.columns:
        if c in CRITERION_COLS:
            out[c] = out[c].astype(int)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

def _predict_with_bundle(bundle, feats_df_row_aligned: pd.DataFrame):
    prep, cal = bundle["prep"], bundle["cal"]
    Xp = prep.transform(feats_df_row_aligned)
    proba = float(cal.predict_proba(Xp)[:, 1][0])
    thr = float(bundle.get("threshold", 0.5))  # tuned threshold support
    pred = int(proba >= thr)
    return pred, proba, thr

# ---------- Suggestions / defaults ----------
def exposure_suggestion(di_val: float) -> str:
    """
    Interpret DI via % deviation: over = 10^(DI/10) - 1
    Tighter bands so mild positives are flagged as overexposed.
    """
    over = (10 ** (di_val / 10.0)) - 1.0  # + = overexposed, − = underexposed
    pct = abs(over) * 100

    if abs(over) < 0.07:
        return "On target → keep technique."
    elif over > 0:
        if pct < 15:
            return f"Slightly overexposed (~+{pct:.0f}%) → ↓ mAs ~10–15%."
        else:
            return f"Overexposed (~+{pct:.0f}%) → ↓ mAs ~15–25%."
    else:
        if pct < 15:
            return f"Slightly underexposed (~−{pct:.0f}%) → ↑ mAs ~10–15%."
        else:
            return f"Underexposed (~−{pct:.0f}%) → ↑ mAs ~15–25%."

def compute_default_eit() -> float:
    if os.path.exists(DATASET_CSV):
        try:
            df = pd.read_csv(DATASET_CSV)
            if "label" in df.columns and "EI_proxy" in df.columns:
                good = df.loc[df["label"] == 1, "EI_proxy"].dropna()
                if len(good): return float(np.median(good.values))
        except Exception:
            pass
    return 120.0

# ---------- Charts (EI to intensity mapping) ----------
def _di_to_ei(eit: float, di: float) -> float:
    return float(eit * (10.0 ** (di / 10.0)))

def _hist_and_cdf(img_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist, edges = np.histogram(img_uint8.ravel(), bins=256, range=(0, 255), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cdf = np.cumsum(hist); cdf = cdf / (cdf[-1] if cdf[-1] != 0 else 1.0)
    return centers, hist, cdf

def _ei_to_intensity(ei_val: float) -> float:
    return 255.0 - float(ei_val)

def fig_exposure_curves(img_uint8: np.ndarray, EI_p: float, EIT: float):
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
    ax1.set_xlabel("Pixel intensity (0–255)"); ax1.set_ylabel("Density")
    ax2 = ax1.twinx(); ax2.plot(centers, cdf, lw=1.2, linestyle="--"); ax2.set_ylabel("CDF")
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
    fig.tight_layout(); return fig

def fig_log_hist(img_uint8: np.ndarray):
    centers, hist, _ = _hist_and_cdf(img_uint8)
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(centers, hist, lw=1.8)
    ax.set_xlabel("Pixel intensity (0–255)"); ax.set_ylabel("Density (log)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(hist[hist>0].min()*0.5, 1e-6))
    fig.tight_layout(); return fig

def fig_regional(img_uint8: np.ndarray):
    h, w = img_uint8.shape[:2]
    x1, x2 = int(w*0.2), int(w*0.8)
    y1, y2 = int(h*0.2), int(h*0.8)
    center = img_uint8[y1:y2, x1:x2]
    mask_c = np.zeros_like(img_uint8, dtype=bool); mask_c[y1:y2, x1:x2] = True
    border = img_uint8[~mask_c]
    hist_c, edges = np.histogram(center.ravel(), bins=256, range=(0,255), density=True)
    hist_b, _     = np.histogram(border.ravel(), bins=256, range=(0,255), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(centers, hist_c, lw=1.6, label="Center (proxy lungs)")
    ax.plot(centers, hist_b, lw=1.0, linestyle="--", label="Borders")
    ax.set_xlabel("Pixel intensity (0–255)"); ax.set_ylabel("Density")
    ax.legend(); fig.tight_layout(); return fig

# ---------- UI ----------
st.set_page_config(page_title="Radiograph QC — Upload & Predict", layout="wide")
st.title("Radiograph QC - Upload & Predict")
st.caption("Prototype sample developed for research discussions with **Dr. Ryan Appleby**.")
st.caption(
    "This prototype is designed for **small animal thoracic radiographs** "
    "(dogs and cats, lateral and ventrodorsal views). "
    "For best results, upload images with the entire thorax visible, "
    "minimal post-processing, and standard exposure technique. "
    "Supports PNG, JPG formats."
)st.caption("EI/DI shown here are relative intensity metrics estimated from the image itself. "
           "They are not the true detector Exposure Index or Deviation Index.")

st.sidebar.header("Models & Target")
exp_bundle  = _load_bundle("exposure_bundle.joblib")
qual_bundle = _load_bundle("quality_bundle.joblib")
st.sidebar.write("Exposure model:", "✅ found" if exp_bundle else "⚠️ not found")
st.sidebar.write("Quality model:",  "✅ found" if qual_bundle else "⚠️ not found")

default_eit = compute_default_eit()
EIT = st.sidebar.number_input("EIT (target, proxy)", value=float(default_eit), step=1.0, format="%.1f")
st.sidebar.caption("Default = median EI_proxy of Good images.")

files = st.file_uploader("Please upload one or more images",
                         type=[e.strip(".") for e in SUPPORTED],
                         accept_multiple_files=True)
if not files:
    st.info("Please upload images to begin."); st.stop()

# # Defaults for criteria (applied to all uploads unless ML overwrites exposure)
# st.sidebar.header("Default criteria (optional)")
# def_sel_map = {"Assume Good": None, "Good (1)": 1, "Poor (0)": 0}
# d_pos = st.sidebar.selectbox("Positioning", list(def_sel_map.keys()), index=0)
# d_exp = st.sidebar.selectbox("Exposure (manual default)", ["Auto from AI","Good (1)","Poor (0)"], index=0)
# d_col = st.sidebar.selectbox("Collimation", list(def_sel_map.keys()), index=0)
# d_shp = st.sidebar.selectbox("Sharpness",   list(def_sel_map.keys()), index=0)
# def _val(sel, auto=False):
#     if auto and sel.startswith("Auto"): return None
#     return def_sel_map.get(sel, None)

# --- Hide Default Criteria Sidebar for now ---
d_pos = None
d_exp = "Auto from AI"
d_col = None
d_shp = None
def _val(sel, auto=False):
    return None

rows = []

for upl in files:
    name = upl.name
    try:
        img = read_image_any_bytes(name, upl.read())
    except Exception as e:
        st.error(f"Cannot read {name}: {e}")
        continue

    # Features & proxies
    qc  = extract_features_from_image(img)
    aux = exposure_aux_features(img)              # sensitive features
    EIp = compute_ei_proxy(img)
    DIp = compute_di_proxy(EIp, EIT)
    ocr = ocr_ei_di_from_image(img)

    feats = {
        "laplacian_var": float(qc["laplacian_var"]),
        "hist_mean": float(qc["hist_mean"]),
        "hist_std": float(qc["hist_std"]),
        "pct_pixels_near_0": float(qc["pct_pixels_near_0"]),
        "pct_pixels_near_255": float(qc["pct_pixels_near_255"]),
        "entropy": float(qc["entropy"]),
        "edge_density": float(qc["edge_density"]),
        "symmetry_score": float(qc["symmetry_score"]),
        "EI_proxy": float(EIp),
        "EIT_in_use": float(EIT),
        "DI_proxy": float(DIp),
        "EI_ocr": float(ocr["EI_ocr"]) if "EI_ocr" in ocr else np.nan,
        "DI_ocr": float(ocr["DI_ocr"]) if "DI_ocr" in ocr else np.nan,
        # dark-tail + percentiles in ROI
        "tail_le_10": float(aux["tail_le_10"]),
        "tail_le_20": float(aux["tail_le_20"]),
        "tail_le_30": float(aux["tail_le_30"]),
        "clip_le_3":  float(aux["clip_le_3"]),
        "p10": float(aux["p10"]),
        "p25": float(aux["p25"]),
        "p50": float(aux["p50"]),
        "p75": float(aux["p75"]),
        "p90": float(aux["p90"]),
    }
    feats_df = pd.DataFrame([feats])

    # Exposure prediction (uses tuned threshold if present)
    e_pred = e_prob = e_thr = None
    if exp_bundle:
        aligned_exp = align_to_bundle_features(exp_bundle, feats_df)
        e_pred, e_prob, e_thr = _predict_with_bundle(exp_bundle, aligned_exp)

        # --- Rule guard (more specific thresholds)
        if e_pred == 1:
            # Overexposure: only if both are high
            if DIp >= 0.30 and feats["tail_le_10"] >= 0.06:
                e_pred = 0
                st.markdown("<hr>", unsafe_allow_html=True)
                st.info("Rule guard: DI ≥ +0.30 and dark-tail≥6% → flagging as overexposed.")
            # Underexposure: only if both are low/bright
            elif DIp <= -0.30 and feats["p90"] > 240:
                e_pred = 0
                st.markdown("<hr>", unsafe_allow_html=True)
                st.info("Rule guard: DI ≤ -0.30 and P90 > 240 → flagging as underexposed.")

    # Fill criterion columns (exposure may come from AI)
    crit_vals = {
        "positioning": _val(d_pos),
        "exposure":    (_val(d_exp, auto=True) if d_exp else None),
        "collimation": _val(d_col),
        "sharpness":   _val(d_shp),
    }
    if crit_vals["exposure"] is None and e_pred is not None:
        crit_vals["exposure"] = int(e_pred)
    for k in ["positioning","collimation","sharpness"]:
        if crit_vals[k] is None: crit_vals[k] = 1
    for k, v in crit_vals.items(): feats_df[k] = int(v)

    # Quality prediction (if bundle present)
    q_pred = q_prob = q_thr = None
    if qual_bundle:
        aligned_qual = align_to_bundle_features(qual_bundle, feats_df)
        q_pred_raw, q_prob, q_thr_tmp = _predict_with_bundle(qual_bundle, aligned_qual)
        q_thr = float(qual_bundle.get("threshold", q_thr_tmp))
        q_pred = int(q_prob >= q_thr)

    # ----------- Render -----------
    # st.markdown("---")
    L, R = st.columns([2,1], gap="large")

    with L:
        st.subheader(name)
        st.image(Image.fromarray(img), use_container_width=True, caption=name)

        m1, m2, m3 = st.columns(3)
        m1.metric("EI_proxy", f"{EIp:.1f}")
        m2.metric("EIT (target)", f"{EIT:.1f}")
        m3.metric("DI_proxy", f"{DIp:+.2f}")
        st.caption(f"% very dark pixels (≤10): {feats['tail_le_10']:.3f} | P25: {feats['p25']:.1f} | P50: {feats['p50']:.1f}")
        # --- Compact exposure interpretation ---
        dark_pct = feats['tail_le_10'] * 100
        p25 = float(feats.get('p25', 0))
        p50 = float(feats.get('p50', 0))

        if dark_pct > 10:
            exp_summary = "Image slightly dark (possible overexposure)."
        elif dark_pct < 3:
            exp_summary = "Image slightly bright (possible underexposure)."
        elif p50 < 90:
            exp_summary = "Image on the darker side."
        elif p50 > 130:
            exp_summary = "Image on the brighter side."
        else:
            exp_summary = "Brightness appears balanced."

        st.markdown(
            f"""
            <div style='margin-top:8px; margin-bottom:10px; padding:6px 10px;
            border-left:3px solid #1f77b4; background-color:#f8fafc; border-radius:6px;'>
            <b>Exposure summary:</b> (≤10): {dark_pct:.1f}% | P25: {p25:.0f} | P50: {p50:.0f}<br>
            <i>{exp_summary}</i>
            </div>
            """,
            unsafe_allow_html=True
        )

        if ocr:
            st.caption(
                "Detected overlays → "
                + ("EI="+str(ocr.get("EI_ocr")) if "EI_ocr" in ocr else "")
                + ("  DI="+str(ocr.get("DI_ocr")) if "DI_ocr" in ocr else "")
            )

        sugg = exposure_suggestion(DIp)
        over = (10 ** (DIp / 10.0)) - 1.0   # + = darker (overexposed), − = brighter (underexposed)
        dev = abs(over)

        if dev < 0.07:
            color = "#16a34a"   # green → on target
        elif dev < 0.15:
            color = "#d97706"   # amber → slightly off
        else:
            color = "#dc2626"   # red → clearly off

        st.markdown(
            f"<div style='padding:10px; border-radius:8px; border:1px solid {color}; "
            f"margin-bottom:12px;'>"
            f"<b>Exposure suggestion:</b> <span style='color:{color}'>{sugg}</span>"
            f"</div>", unsafe_allow_html=True
        )

        with st.expander("📊 Exposure curves (Histogram + CDF)", expanded=True):
            st.pyplot(fig_exposure_curves(img, EIp, EIT), use_container_width=True)
        with st.expander("📈 Log-density histogram (tail visibility)", expanded=False):
            st.pyplot(fig_log_hist(img), use_container_width=True)
        with st.expander("🗺️ Regional comparison (center vs borders)", expanded=False):
            st.pyplot(fig_regional(img), use_container_width=True)

    with R:
        st.subheader("AI predictions")
        if exp_bundle:
            # Badge now follows FINAL decision (after any rule guard)
            exp_status = 'Good' if e_pred == 1 else 'Poor'
            badge = '🟢' if e_pred == 1 else '🔴'
            st.metric("Exposure", f"{badge} {exp_status}",
                      f"p={e_prob:.2f} | thr={e_thr:.2f}")
        else:
            st.info("Exposure model not found. Train first.")

        if qual_bundle:
            badge_q = "🟢" if q_prob is not None and q_prob >= q_thr else "🔴"
            st.metric("Final Quality", f"{badge_q} {'Good' if q_pred==1 else 'Poor'}",
                      f"p={q_prob:.2f} | thr={q_thr:.2f}" if q_prob is not None else "no prob")
        else:
            st.info("Quality model not found.")

        st.markdown("### Quick summary")
        bullets = [
            f"DI_proxy {DIp:+.2f} → {sugg}",
            f"Dark-tail≤10: {feats['tail_le_10']:.3f}",
            f"P25/P50: {feats['p25']:.1f}/{feats['p50']:.1f}",
        ]
        if exp_bundle:
            bullets.append(f"AI Exposure: **{'Good' if e_pred==1 else 'Poor'}** (p={e_prob:.2f}, thr={e_thr:.2f})")
        if qual_bundle:
            bullets.append(f"AI Final Quality: **{'Good' if q_pred==1 else 'Poor'}** (p={q_prob:.2f}, thr={q_thr:.2f})")
        bullets.append(f"Criteria used: pos={crit_vals['positioning']}, exp={crit_vals['exposure']}, "
                       f"col={crit_vals['collimation']}, shp={crit_vals['sharpness']}")
        st.markdown("- " + "\n- ".join(bullets))

    rows.append({
        "image_name": name,
        "EI_proxy": EIp, "EIT": EIT, "DI_proxy": DIp,
        "tail_le_10": feats["tail_le_10"], "tail_le_20": feats["tail_le_20"],
        "tail_le_30": feats["tail_le_30"], "clip_le_3": feats["clip_le_3"],
        "p10": feats["p10"], "p25": feats["p25"], "p50": feats["p50"],
        "p75": feats["p75"], "p90": feats["p90"],
        "AI_exposure": (None if e_pred is None else int(e_pred)),
        "AI_exposure_p": (None if e_prob is None else float(e_prob)),
        "AI_exposure_thr": (None if e_thr is None else float(e_thr)),
        "AI_quality": (None if q_pred is None else int(q_pred)),
        "AI_quality_p": (None if q_prob is None else float(q_prob)),
        "AI_quality_thr": (None if q_thr is None else float(q_thr)),
        "positioning": int(crit_vals["positioning"]),
        "exposure_criterion": int(crit_vals["exposure"]),
        "collimation": int(crit_vals["collimation"]),
        "sharpness": int(crit_vals["sharpness"]),
        "suggestion": sugg
    })

st.markdown("## Batch summary")
summary_df = pd.DataFrame(rows)
st.dataframe(summary_df, use_container_width=True)
csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download predictions CSV", data=csv, file_name="qc_predictions.csv", mime="text/csv")

st.caption("Note: EI/DI are proxies for PNGs.")
