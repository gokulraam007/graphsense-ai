# ===============================================
# GraphSense-AI v5.1: LLM-Assisted Data Extractor
# CV for accuracy | LLM for intelligence
# ===============================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
import io
import openai
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("📊 GraphSense-AI v5.1 LLM-Assisted Data Extractor")

# ===============================================
# USER INPUT (LLM CONTEXT)
# ===============================================
st.sidebar.header("🧠 Graph Context (LLM)")
graph_type = st.sidebar.selectbox(
    "Graph Type", 
    ["IV Curve (Solar Cell)", "Light Transmission", "Generic X-Y Curve"]
)
x_unit = st.sidebar.text_input("X-axis Unit", "Voltage (V)")
y_unit = st.sidebar.text_input("Y-axis Unit", "Current (A)")

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

def create_excel_export(points):
    df = pd.DataFrame(points, columns=["Pixel_X", "Pixel_Y"])
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
    Assess: (1) Point density? (2) Curve smooth or noisy? (3) Risks? Answer in 2-3 sentences."""
    
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
st.write("Extract accurate data points using Computer Vision + LLM intelligence")

uploaded = st.file_uploader("Upload Graph Image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Graph", use_container_width=True)
    
    st.subheader("Extraction Settings")
    col1, col2 = st.columns(2)
    with col1:
        blur = st.slider("Blur Strength", 1, 15, 5)
    with col2:
        threshold = st.slider("Detection Threshold", 50, 250, 200)
    
    if st.button("Extract Data Points", use_container_width=True):
        with st.spinner("Processing..."):
            mask = detect_curve_points(image, blur, threshold)
            points = extract_data_points(mask)
        
        if len(points) == 0:
            st.error("No curve detected")
        else:
            st.success(f"Extracted {len(points)} points")
            
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.imshow(mask, cmap="gray")
                ax1.set_title("Curve Mask")
                st.pyplot(fig1)
            with col2:
                fig2, ax2 = plt.subplots()
                ax2.scatter(points[:, 0], points[:, 1], s=2)
                ax2.set_title("Data Points")
                st.pyplot(fig2)
            
            st.subheader("Data Preview")
            df_preview = pd.DataFrame(points[:20], columns=['Pixel_X', 'Pixel_Y'])
            st.dataframe(df_preview)
            
            st.subheader("LLM Assessment")
            feedback = llm_quality_analysis(points, graph_type)
            st.info(feedback)
            
            st.subheader("Export")
            col1, col2 = st.columns(2)
            with col1:
                excel_buffer = create_excel_export(points)
                st.download_button("Download Excel", data=excel_buffer, file_name="graph_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with col2:
                csv_data = pd.DataFrame(points, columns=['Pixel_X', 'Pixel_Y']).to_csv(index=False)
                st.download_button("Download CSV", data=csv_data, file_name="graph_data.csv", mime="text/csv", use_container_width=True)
