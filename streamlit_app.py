# ==========================================================
# GraphSense-AI : Physics-aware Graph Digitizer (Streamlit)
# FIXED VERSION - No cv2 dependency (Streamlit Cloud compatible)
# ==========================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
from functools import lru_cache
from scipy.optimize import curve_fit
from paddleocr import PaddleOCR
from scipy import ndimage

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
# BINARY IMAGE FROM PIL
# =============================
def image_to_binary_mask(image, threshold=200):
    """Convert PIL image to binary mask using PIL only"""
    # Convert to grayscale
    gray = image.convert('L')
    # Convert to numpy array
    gray_arr = np.array(gray)
    # Create binary mask
    mask = (gray_arr < threshold).astype(np.uint8) * 255
    return mask

# =============================
# MASK CLEANING (scipy-based)
# =============================
def clean_curve_mask(mask, min_area=150):
    """Clean mask using scipy.ndimage instead of cv2"""
    from scipy import ndimage
    labeled, num_features = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(num_features + 1))
    
    cleaned = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        if sizes[i] >= min_area:
            cleaned[labeled == i] = 255
    
    return cleaned.astype(np.uint8)

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
    if np.sum(mask) < 2:
        return lambda v: np.zeros_like(v)
    
    try:
        popt, _ = curve_fit(
            iv_equation,
            V[mask],
            I[mask],
            p0=[np.max(I), np.max(V)],
            maxfev=10000
        )
    except:
        return lambda v: np.zeros_like(v)

    def model(v):
        return np.clip(iv_equation(v, *popt), 0, None)

    return model

# =============================
# VALIDATION
# =============================
def validate_iv_curve(V, I):
    if len(I) == 0 or len(V) == 0:
        return {}, {}
    
    Isc = float(I[0]) if len(I) > 0 else 0
    Voc = float(V[np.where(I <= 0)[0][0]]) if any(I <= 0) else float(V[-1]) if len(V) > 0 else 0
    Pmax = float(np.max(V * I)) if len(V) > 0 and len(I) > 0 else 0

    metrics = {"Isc": Isc, "Voc": Voc, "Pmax": Pmax}
    checks = {
        "monotonic_current": all(I[i] >= I[i+1] for i in range(len(I)-1)) if len(I) > 1 else True,
        "positive_power": Pmax > 0
    }
    return metrics, checks

def generate_quality_report(metrics, checks):
    if len(checks) == 0:
        return {"quality_score": 0, "flags": ["No data"], "recommendation": "REJECT"}
    
    score = sum(checks.values()) / len(checks)
    return {
        "quality_score": round(score, 2),
        "flags": [k for k, v in checks.items() if not v],
        "recommendation": "REVIEW" if score < 0.8 else "ACCEPT"
    }

# =============================
# STREAMLIT UI
# =============================
st.subheader("📊 Upload Your Graph Image")
uploaded = st.file_uploader(
    "Upload IV Curve or Transmission Graph (PNG, JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Graph", use_container_width=True)

    st.info("🔄 Processing... Extracting OCR labels and fitting curve")

    # Convert image to binary mask
    mask = image_to_binary_mask(image, threshold=200)
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
        
        if len(curve_pixels) > 0:
            V_raw = pixel_to_value(x_model, curve_pixels[:, 0])
            I_raw = pixel_to_value(y_model, curve_pixels[:, 1])

            iv_model = fit_iv_curve(V_raw, I_raw)
            V_fit = np.linspace(min(V_raw), max(V_raw), 300)
            I_fit = iv_model(V_fit)

            metrics, checks = validate_iv_curve(V_fit, I_fit)
            report = generate_quality_report(metrics, checks)

            # VISUALIZATION
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Extracted Curve Mask")
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                ax1.imshow(cleaned_mask, cmap="gray")
                ax1.set_title("Binary Mask - Detected Curve")
                ax1.set_xlabel("Pixel X")
                ax1.set_ylabel("Pixel Y")
                st.pyplot(fig1)
            
            with col2:
                st.subheader("IV Curve Fit")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.plot(V_fit, I_fit, "b-", linewidth=2, label="Physics Fit")
                ax2.scatter(V_raw, I_raw, s=10, alpha=0.3, label="Extracted Data")
                ax2.set_xlabel("Voltage (V)")
                ax2.set_ylabel("Current (A)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            
            st.subheader("📋 Quality Report")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Quality Score", f"{report['quality_score']}", "Pass" if report['recommendation'] == "ACCEPT" else "Review")
            with col2:
                st.metric("Isc (A)", f"{metrics.get('Isc', 0):.3f}")
            with col3:
                st.metric("Voc (V)", f"{metrics.get('Voc', 0):.3f}")
            
            if report['flags']:
                st.warning(f"⚠ Issues detected: {', '.join(report['flags'])}")
            
            st.json(report)
        else:
            st.error("No curve pixels detected in the image. Try adjusting the image contrast.")
    else:
        st.error(f"OCR could not detect enough axis labels. Found X-axis: {len(ticks_x)}, Y-axis: {len(ticks_y)}")
else:
    st.info("👆 Please upload a graph image to begin processing")
