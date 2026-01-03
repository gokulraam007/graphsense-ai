# ==========================================================
# GraphSense-AI : Physics-aware Graph Digitizer (Streamlit)
# VERSION 4.0 - Manual Label Entry + Robust Error Handling
# ==========================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
from functools import lru_cache
from scipy.optimize import curve_fit
from scipy import ndimage
from paddleocr import PaddleOCR
import warnings
warnings.filterwarnings('ignore')

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
    try:
        return PaddleOCR(use_angle_cls=True, lang='en')
    except Exception as e:
        st.error(f"Failed to initialize OCR model: {str(e)}")
        return None

# =============================
# OCR LABEL EXTRACTION
# =============================
def extract_numeric_labels_with_positions(image_pil, axis="x"):
    try:
        image_array = np.array(image_pil)
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array]*3, axis=-1)
        elif image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        ocr = get_ocr_model()
        if ocr is None:
            return []
        result = ocr.ocr(image_array)
        if result is None or len(result) == 0:
            return []
        items = []
        for line in result:
            if line is None:
                continue
            for word_info in line:
                try:
                    bbox = word_info[0]
                    text = word_info[1][0]
                    confidence = word_info[1][1]
                    if confidence < 0.3:
                        continue
                    match = re.search(r"-?\d+(\.\d+)?", text)
                    if not match:
                        continue
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    pixel_pos = np.mean(xs) if axis == "x" else np.mean(ys)
                    value = float(match.group())
                    items.append((pixel_pos, value))
                except (IndexError, ValueError, AttributeError):
                    continue
        return sorted(items, key=lambda x: x[0])
    except Exception as e:
        st.warning(f"OCR processing: {type(e).__name__}: {str(e)}")
        return []

# =============================
# BINARY IMAGE FROM PIL
# =============================
def image_to_binary_mask(image_pil, threshold=200):
    try:
        gray = image_pil.convert('L')
        gray_arr = np.array(gray)
        mask = (gray_arr < threshold).astype(np.uint8) * 255
        return mask
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return np.zeros((image_pil.size[1], image_pil.size[0]), dtype=np.uint8)

# =============================
# MASK CLEANING
# =============================
def clean_curve_mask(mask, min_area=150):
    try:
        labeled, num_features = ndimage.label(mask)
        sizes = ndimage.sum(mask, labeled, range(num_features + 1))
        cleaned = np.zeros_like(mask)
        for i in range(1, num_features + 1):
            if i < len(sizes) and sizes[i] >= min_area:
                cleaned[labeled == i] = 255
        return cleaned.astype(np.uint8)
    except Exception as e:
        st.warning(f"Mask cleaning: {str(e)}")
        return mask

# =============================
# CURVE PIXEL EXTRACTION
# =============================
def extract_curve_pixels(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([])
    return np.column_stack((xs, ys))

# =============================
# AXIS CALIBRATION
# =============================
def fit_axis_calibration(pixels, values):
    try:
        if len(pixels) < 2:
            return [1, 0]
        return np.polyfit(pixels, values, 1)
    except:
        return [1, 0]

def pixel_to_value(model, pixels):
    return model[0] * np.array(pixels) + model[1]

# =============================
# PHYSICS-AWARE IV MODEL
# =============================
def iv_equation(V, Isc, Voc):
    return Isc * (1 - V / Voc)

def fit_iv_curve(V, I):
    try:
        mask = (V >= 0) & (I >= 0)
        if np.sum(mask) < 2:
            return lambda v: np.zeros_like(v)
        Isc_init = np.max(I[mask]) if np.max(I[mask]) > 0 else 1
        Voc_init = np.max(V[mask]) if np.max(V[mask]) > 0 else 1
        popt, _ = curve_fit(iv_equation, V[mask], I[mask], p0=[Isc_init, Voc_init], maxfev=10000, ftol=1e-6)
        def model(v):
            return np.clip(iv_equation(v, *popt), 0, None)
        return model
    except:
        return lambda v: np.zeros_like(v)

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
    checks = {"monotonic_current": all(I[i] >= I[i+1] for i in range(len(I)-1)) if len(I) > 1 else True, "positive_power": Pmax > 0}
    return metrics, checks

def generate_quality_report(metrics, checks):
    if len(checks) == 0:
        return {"quality_score": 0, "flags": ["No data"], "recommendation": "REJECT"}
    score = sum(checks.values()) / len(checks)
    return {"quality_score": round(score, 2), "flags": [k for k, v in checks.items() if not v], "recommendation": "REVIEW" if score < 0.8 else "ACCEPT"}

# =============================
# STREAMLIT UI
# =============================
st.subheader("📊 Upload Your Graph Image")
uploaded = st.file_uploader("Upload IV Curve or Transmission Graph (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Graph", use_container_width=True)
    st.info("🔄 Processing... Extracting OCR labels and fitting curve")

    mask = image_to_binary_mask(image, threshold=200)
    cleaned_mask = clean_curve_mask(mask)
    ticks_x = extract_numeric_labels_with_positions(image, axis="x")
    ticks_y = extract_numeric_labels_with_positions(image, axis="y")
    st.info(f"📊 Detected labels - X-axis: {len(ticks_x)}, Y-axis: {len(ticks_y)}")

    # NEW: Manual Entry if OCR Detection is Poor
    if len(ticks_x) < 2 or len(ticks_y) < 2:
        st.warning("⚠️ OCR could not detect enough labels. Please enter them manually to continue.")
        with st.expander("🗒️ Manual Label Entry", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("X-Axis Labels")
                x_labels_input = st.text_input("Enter X-axis values (comma-separated):", placeholder="200, 400, 600, 800")
            with col2:
                st.subheader("Y-Axis Labels")
                y_labels_input = st.text_input("Enter Y-axis values (comma-separated):", placeholder="-26.523, 0, 50")
            
            if st.button("✅ Load Manual Labels"):
                if x_labels_input and y_labels_input:
                    try:
                        x_values = [float(x.strip()) for x in x_labels_input.split(',')]
                        y_values = [float(y.strip()) for y in y_labels_input.split(',')]
                        img_width, img_height = image.size
                        x_pixels = np.linspace(50, img_width - 50, len(x_values))
                        y_pixels = np.linspace(img_height - 50, 50, len(y_values))
                        ticks_x = list(zip(x_pixels, x_values))
                        ticks_y = list(zip(y_pixels, y_values))
                        st.success("✅ Manual labels loaded!")
                    except ValueError:
                        st.error("❌ Invalid input. Please enter numbers separated by commas.")
                else:
                    st.error("❌ Please fill in both X and Y axis labels.")

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
                ax2.set_xlabel("Voltage")
                ax2.set_ylabel("Current")
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
                st.warning(f"⚠ Issues: {', '.join(report['flags'])}")
            st.json(report)
        else:
            st.error("No curve pixels detected. Try adjusting image contrast.")
else:
    st.info("👆 Upload a graph image to begin processing")
