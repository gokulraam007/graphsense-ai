# ===============================================
# GraphSense-AI v6.0: Advanced LLM-Assisted Data Extractor
# CV for accuracy | LLM for intelligence | Pixel-to-Data conversion
# ===============================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from scipy import ndimage
import io
import openai
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📊 GraphSense-AI v6.0 Advanced Data Extractor")

# ===============================================
# USER INPUT (LLM CONTEXT)
# ===============================================
st.sidebar.header("🧠 Graph Context (LLM)")
graph_type = st.sidebar.selectbox(
    "Graph Type", 
    ["IV Curve (Solar Cell)", "Light Transmission", "Spectral Response", "Generic X-Y Curve"]
)
x_unit = st.sidebar.text_input("X-axis Unit", "Voltage (V)")
y_unit = st.sidebar.text_input("Y-axis Unit", "Current (A)")

# ===============================================
# AXIS DETECTION & COORDINATE CONVERSION
# ===============================================
def detect_axis_bounds(image_pil):
    """Detect axis boundaries from the image"""
    gray = np.array(image_pil.convert("L"))
    height, width = gray.shape
    
    # Find black pixels (axes lines, typically very dark)
    threshold = 50
    dark_pixels = gray < threshold
    
    # Find axis extents
    rows_with_pixels = np.any(dark_pixels, axis=1)
    cols_with_pixels = np.any(dark_pixels, axis=0)
    
    y_indices = np.where(rows_with_pixels)[0]
    x_indices = np.where(cols_with_pixels)[0]
    
    if len(y_indices) > 0 and len(x_indices) > 0:
        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]
    else:
        y_min, y_max = 0, height
        x_min, x_max = 0, width
    
    return x_min, x_max, y_min, y_max, width, height

def extract_axis_labels(image_pil):
    """Extract X and Y axis range values from image text"""
    # This would require OCR - for now return default
    # In production, use Tesseract or Claude vision
    return None, None, None, None

def pixel_to_data(pixel_points, image_pil, x_min_data=None, x_max_data=None, 
                  y_min_data=None, y_max_data=None):
    """Convert pixel coordinates to actual data values"""
    if len(pixel_points) == 0:
        return np.array([])
    
    x_min_px, x_max_px, y_min_px, y_max_px, width, height = detect_axis_bounds(image_pil)
    
    # If data bounds not provided, estimate from pixel bounds
    if x_min_data is None:
        x_min_data = 0
        x_max_data = 1000  # Default - should be detected from axis labels
    
    if y_min_data is None:
        y_min_data = 0
        y_max_data = 100  # Default - should be detected from axis labels
    
    # Extract pixel coordinates
    px_x = pixel_points[:, 0]
    px_y = pixel_points[:, 1]
    
    # Normalize pixel coordinates to [0, 1]
    norm_x = (px_x - x_min_px) / (x_max_px - x_min_px + 1e-6)
    norm_y = (px_y - y_min_px) / (y_max_px - y_min_px + 1e-6)
    
    # Clip to valid range
    norm_x = np.clip(norm_x, 0, 1)
    norm_y = np.clip(norm_y, 0, 1)
    
    # Convert to data coordinates (Y is inverted in images)
    data_x = x_min_data + norm_x * (x_max_data - x_min_data)
    data_y = y_max_data - norm_y * (y_max_data - y_min_data)  # Inverted Y
    
    return np.column_stack((data_x, data_y))

# ===============================================
# IMAGE PROCESSING
# ===============================================
def detect_curve_points(image_pil, blur_strength=5, threshold=200):
    gray = np.array(image_pil.convert("L"))
    blurred = gaussian_filter(gray.astype(float), sigma=blur_strength)
    mask = (blurred < threshold)
    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=2)
    return (mask * 255).astype(np.uint8)

def extract_data_points(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([])
    points = np.column_stack((xs, ys))
    return points[np.argsort(points[:, 0])]

def smooth_curve(points, window_size=5):
    """Apply moving average smoothing to reduce noise"""
    if len(points) < window_size:
        return points
    
    smoothed = np.zeros_like(points)
    for i in range(len(points)):
        start = max(0, i - window_size // 2)
        end = min(len(points), i + window_size // 2 + 1)
        smoothed[i] = np.mean(points[start:end], axis=0)
    
    return smoothed

def remove_duplicates(points, tolerance=2):
    """Remove duplicate/very close points"""
    if len(points) < 2:
        return points
    
    unique_points = [points[0]]
    for point in points[1:]:
        dist = np.sqrt((point[0] - unique_points[-1][0])**2 + 
                       (point[1] - unique_points[-1][1])**2)
        if dist > tolerance:
            unique_points.append(point)
    
    return np.array(unique_points)

def create_excel_export(pixel_points, data_points):
    """Create Excel with both pixel and data coordinates"""
    df = pd.DataFrame({
        "Pixel_X": pixel_points[:, 0].astype(int),
        "Pixel_Y": pixel_points[:, 1].astype(int),
        "Data_X": data_points[:, 0],
        "Data_Y": data_points[:, 1]
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Extracted_Data")
    buffer.seek(0)
    return buffer

# ===============================================
# LLM ANALYSIS (OPENAI)
# ===============================================
def llm_quality_analysis(points, graph_type):
    if len(points) < 50:
        return "Too few points detected."
    
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        return "OpenAI API key not configured. Add to Streamlit secrets."
    
    prompt = f"""Scientific data assistant. Graph: {graph_type}. Points: {len(points)}. 
    Assess: (1) Point density? (2) Curve smooth or noisy? (3) Data quality risks? Answer in 2-3 sentences."""
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM analysis unavailable: {str(e)}"

# ===============================================
# STREAMLIT UI
# ===============================================
st.write("Extract accurate data points using Computer Vision + LLM intelligence + Pixel-to-Data conversion")
uploaded = st.file_uploader("Upload Graph Image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Graph", use_container_width=True)
    
    st.subheader("Extraction Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        blur = st.slider("Blur Strength", 1, 15, 5)
    with col2:
        threshold = st.slider("Detection Threshold", 50, 250, 200)
    with col3:
        smooth = st.checkbox("Apply Smoothing", value=True)
    
    col4, col5 = st.columns(2)
    with col4:
        x_min = st.number_input("X-axis Min Value", value=0.0)
    with col5:
        x_max = st.number_input("X-axis Max Value", value=1000.0)
    
    col6, col7 = st.columns(2)
    with col6:
        y_min = st.number_input("Y-axis Min Value", value=0.0)
    with col7:
        y_max = st.number_input("Y-axis Max Value", value=100.0)
    
    if st.button("Extract Data Points", use_container_width=True):
        with st.spinner("Processing..."):
            mask = detect_curve_points(image, blur, threshold)
            pixel_points = extract_data_points(mask)
            
            if len(pixel_points) == 0:
                st.error("No curve detected")
            else:
                # Remove duplicates and apply smoothing
                pixel_points = remove_duplicates(pixel_points)
                if smooth:
                    pixel_points = smooth_curve(pixel_points, window_size=3)
                
                # Convert to data coordinates
                data_points = pixel_to_data(pixel_points, image, x_min, x_max, y_min, y_max)
                
                st.success(f"Extracted {len(pixel_points)} points")
                
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(mask, cmap="gray")
                    ax1.set_title("Curve Mask")
                    st.pyplot(fig1)
                with col2:
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(data_points[:, 0], data_points[:, 1], s=5, alpha=0.6)
                    ax2.set_xlabel(x_unit)
                    ax2.set_ylabel(y_unit)
                    ax2.set_title("Extracted Data Points")
                    st.pyplot(fig2)
                
                st.subheader("Data Preview")
                df_preview = pd.DataFrame({
                    'Pixel_X': pixel_points[:20, 0].astype(int),
                    'Pixel_Y': pixel_points[:20, 1].astype(int),
                    'Data_X': data_points[:20, 0],
                    'Data_Y': data_points[:20, 1]
                })
                st.dataframe(df_preview)
                
                st.subheader("Data Statistics")
                col_stats1, col_stats2 = st.columns(2)
                with col_stats1:
                    st.metric("Total Points", len(data_points))
                    st.metric("X Range", f"{data_points[:, 0].min():.2f} to {data_points[:, 0].max():.2f}")
                with col_stats2:
                    st.metric("Y Range", f"{data_points[:, 1].min():.2f} to {data_points[:, 1].max():.2f}")
                
                st.subheader("LLM Assessment")
                feedback = llm_quality_analysis(data_points, graph_type)
                st.info(feedback)
                
                st.subheader("Export")
                col1, col2 = st.columns(2)
                with col1:
                    excel_buffer = create_excel_export(pixel_points, data_points)
                    st.download_button("Download Excel", data=excel_buffer, file_name="graph_data.xlsx", 
                                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                                     use_container_width=True)
                with col2:
                    csv_data = pd.DataFrame({
                        'Data_X': data_points[:, 0],
                        'Data_Y': data_points[:, 1]
                    }).to_csv(index=False)
                    st.download_button("Download CSV (Data Only)", data=csv_data, file_name="graph_data.csv", 
                                     mime="text/csv", use_container_width=True)
