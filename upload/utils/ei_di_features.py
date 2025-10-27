# utils/ei_di_features.py
import numpy as np
from typing import Dict, Tuple

# -------------------
# ROI (coarse lung crop)
# -------------------
def _center_lung_roi(img_u8: np.ndarray, x_margin=0.10, y_margin=0.10) -> np.ndarray:
    """
    Remove ~10% from each side to avoid collimation glare/labels.
    Works for both small animals and humans; keeps the thorax center.
    """
    h, w = img_u8.shape[:2]
    x1, x2 = int(w * x_margin), int(w * (1 - x_margin))
    y1, y2 = int(h * y_margin), int(h * (1 - y_margin))
    roi = img_u8[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    return roi if roi.size else img_u8

def _percentiles(x: np.ndarray, qs=(10, 25, 50, 75, 90)) -> Dict[str, float]:
    p = np.percentile(x, qs)
    return {f"p{q}": float(v) for q, v in zip(qs, p)}

def _tail_masses(x: np.ndarray) -> Dict[str, float]:
    n = max(1, x.size)
    return {
        "tail_le_10": float((x <= 10).sum()) / n,  # very dark
        "tail_le_20": float((x <= 20).sum()) / n,  # dark
        "tail_le_30": float((x <= 30).sum()) / n,  # mid-dark
        "clip_le_3":  float((x <= 3 ).sum()) / n,  # near-black clipping
    }

# -------------------
# Exposure indices
# -------------------
def compute_ei_proxy(img_u8: np.ndarray) -> float:
    """
    More sensitive EI proxy (higher when exposure is higher = darker image).
    Uses low-percentiles in a lung-biased ROI instead of mean:
      EI_proxy = 255 - (0.6*P25 + 0.4*P50)_ROI
    """
    roi = _center_lung_roi(img_u8, x_margin=0.10, y_margin=0.10)
    flat = roi.ravel().astype(np.float32)
    if flat.size == 0:
        return 0.0
    p = _percentiles(flat, (25, 50))
    intensity_robust = 0.6 * p["p25"] + 0.4 * p["p50"]
    return float(255.0 - intensity_robust)

def compute_di_proxy(EI: float, EIT: float, eps: float = 1e-6) -> float:
    """
    DI = 10 * log10(EI / EIT)  (positive = overexposed; negative = underexposed)
    """
    EI  = max(float(EI),  eps)
    EIT = max(float(EIT), eps)
    return 10.0 * np.log10(EI / EIT)

def exposure_aux_features(img_u8: np.ndarray) -> Dict[str, float]:
    """
    Extra exposure-sensitivity features (feed to model and rules):
      - tail masses (≤10/20/30), black clipping ≤3
      - percentiles P10..P90 inside ROI
    """
    roi = _center_lung_roi(img_u8, x_margin=0.10, y_margin=0.10)
    flat = roi.ravel().astype(np.float32)
    feats = _tail_masses(flat)
    feats.update(_percentiles(flat, (10, 25, 50, 75, 90)))
    return feats

def ocr_ei_di_from_image(img_u8: np.ndarray) -> Dict[str, float]:
    """
    Optional: add your console-overlay OCR here.
    Example return: {"EI_ocr": 210.0, "DI_ocr": 0.3}
    """
    return {}