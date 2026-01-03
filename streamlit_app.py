# ==========================================================
# GraphSense-AI : Physics-aware Graph Digitizer (Streamlit)
# Author: You
# ==========================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
try:
    import cv2
except ImportError as e:
    st.error(f"OpenCV (cv2) is not available: {e}")
    cv2 = None
import re
from PIL import Image
from functools import lru_cache
from scipy.optimize import curve_fit
from paddleocr import PaddleOCR

# =============================
# CONFIG
# =============================
st.set_page_config(layout="wide")
st.title("GraphSense-AI | Physics-Aware Graph Digitizer")

# =============================
# OCR ENGINE (CACHED)
# =============================
@lru_cache(maxsize=1)
def get_ocr_model():
    return PaddleOCR(use_angle_cls=True, lang="en")

# =============================
# OCR LABEL EXTRACTION
# =============================
def extract_numeric_labels_with_positions(image, axis="x"):
    ocr = get_ocr_model()
    result = ocr.ocr(np.array(image), cls=True)
    items = []

    for line in result:
        for word in line:
            text = word[1][0]
            bbox = word[0]

            match = re.search(r"-?\d+(\.\d+)?", text)
            if not match:
                continue

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]

            pixel_pos = np.mean(xs) if axis == "x" else np.mean(ys)
            items.append((pixel_pos, float(match.group())))

    return sorted(items, key=lambda x: x[0])

# =============================
# MASK CLEANING
# =============================
def clean_curve_mask(mask, min_area=150):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned

# =============================
# CURVE PIXEL EXTRACTION
# =============================
def extract_curve_pixels(mask):
    ys, xs = np.where(mask > 0)
    return np.column_stack((xs, ys))

# =============================
# AXIS CALIBRATION
# =============================
def fit_axis_calibration(pixels, values):
    return np.polyfit(pixels, values, 1)

def pixel_to_value(model, pixels):
    return model[0] * np.array(pixels) + model[1]

# =============================
# PHYSICS-AWARE IV MODEL
# =============================
def iv_equation(V, Isc, Voc):
    return Isc * (1 - V / Voc)

def fit_iv_curve(V, I):
    mask = (V >= 0) & (I >= 0)
    popt, _ = curve_fit(
        iv_equation,
        V[mask],
        I[mask],
        p0=[np.max(I), np.max(V)]
    )

    def model(v):
        return np.clip(iv_equation(v, *popt), 0, None)

    return model

# =============================
# VALIDATION
# =============================
def validate_iv_curve(V, I):
    Isc = float(I[0])
    Voc = float(V[np.where(I <= 0)[0][0]]) if any(I <= 0) else float(V[-1])
    Pmax = float(np.max(V * I))

    metrics = {"Isc": Isc, "Voc": Voc, "Pmax": Pmax}
    checks = {
        "monotonic_current": all(I[i] >= I[i+1] for i in range(len(I)-1)),
        "positive_power": Pmax > 0
    }
    return metrics, checks

def generate_quality_report(metrics, checks):
    score = sum(checks.values()) / len(checks)
    return {
        "quality_score": round(score, 2),
        "flags": [k for k, v in checks.items() if not v],
        "recommendation": "REVIEW" if score < 0.8 else "ACCEPT"
    }

# =============================
# STREAMLIT UI
# =============================
uploaded = st.file_uploader(
    "Upload IV Curve or Transmission Graph",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Graph", use_container_width=True)

    st.warning("⚠ YOLOv8 auto-detection placeholder – demo curve mask used")

    # -----------------------------
    # DEMO CURVE MASK (SAFE DEFAULT)
    # -----------------------------
    mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
    h = mask.shape[0]
    for x in range(50, mask.shape[1] - 50):
        y = int(h * (1 - x / mask.shape[1]))
        mask[y, x] = 255

    cleaned_mask = clean_curve_mask(mask)

    # OCR AXIS TICKS
    ticks_x = extract_numeric_labels_with_positions(image, axis="x")
    ticks_y = extract_numeric_labels_with_positions(image, axis="y")

    if len(ticks_x) >= 2 and len(ticks_y) >= 2:
        px_x, val_x = zip(*ticks_x)
        px_y, val_y = zip(*ticks_y)

        x_model = fit_axis_calibration(px_x, val_x)
        y_model = fit_axis_calibration(px_y, val_y)

        curve_pixels = extract_curve_pixels(cleaned_mask)
        V_raw = pixel_to_value(x_model, curve_pixels[:, 0])
        I_raw = pixel_to_value(y_model, curve_pixels[:, 1])

        iv_model = fit_iv_curve(V_raw, I_raw)
        V_fit = np.linspace(min(V_raw), max(V_raw), 300)
        I_fit = iv_model(V_fit)

        metrics, checks = validate_iv_curve(V_fit, I_fit)
        report = generate_quality_report(metrics, checks)

        # -----------------------------
        # DEBUG VISUALIZATION
        # -----------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.imshow(cleaned_mask, cmap="gray")
        ax1.set_title("Cleaned Curve Mask")

        ax2.plot(V_fit, I_fit, "b-", label="Physics Fit")
        ax2.scatter(V_raw, I_raw, s=5, alpha=0.3, label="Extracted")
        ax2.set_xlabel("Voltage (V)")
        ax2.set_ylabel("Current (A)")
        ax2.legend()

        st.pyplot(fig)

        st.subheader("Quality Report")
        st.json(report)

    else:
        st.error("OCR could not reliably detect axis labels.")
