# utils/features.py
import numpy as np
import cv2
from skimage.measure import shannon_entropy

def extract_features_from_image(img: np.ndarray) -> dict:
    """
    img: 2D grayscale array (uint8 or convertible)
    returns: dict of simple, robust QC features
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # normalize to 0..255 uint8
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # sharpness
    lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())

    # histogram-based exposure metrics
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    hist /= (hist.sum() + 1e-8)
    px = np.arange(256, dtype=np.float32)
    mean = float((hist * px).sum())
    std = float(np.sqrt((hist * (px - mean) ** 2).sum()))
    pct_dark = float(hist[:8].sum())        # near black
    pct_bright = float(hist[-8:].sum())     # near white

    # texture/noise proxy
    entropy = float(shannon_entropy(img))

    # edge density
    edges = cv2.Canny(img, 50, 150)
    edge_density = float(edges.mean())

    # crude left-right symmetry (rotation cue)
    h, w = img.shape[:2]
    left = img[:, :w // 2].astype(np.float32)
    right = np.fliplr(img[:, w - w // 2:].astype(np.float32))
    symmetry = 1.0 - float(np.mean(np.abs(left - right)) / 255.0)

    return {
        "laplacian_var": lap_var,
        "hist_mean": mean,
        "hist_std": std,
        "pct_pixels_near_0": pct_dark,
        "pct_pixels_near_255": pct_bright,
        "entropy": entropy,
        "edge_density": edge_density,
        "symmetry_score": symmetry,
    }