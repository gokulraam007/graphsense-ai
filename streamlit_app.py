# ===============================================
# GraphSense-AI v5.0: Data Point Extractor
# Skip OCR, Focus on Accurate Data Extraction
# Export to Excel
# ===============================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
st.title("📊 GraphSense-AI v5.0 - Data Point Extractor")
st.subheader("Extract graph data points accurately and export to Excel")

# ===============================================
# IMAGE PROCESSING FOR CURVE DETECTION
# ===============================================

def detect_curve_points(image_pil, blur_strength=5, threshold=200):
    """
    Detect curve/line pixels in the graph using edge detection
    and morphological operations
    """
    image_array = np.array(image_pil.convert('L'))
    
    # Apply Gaussian blur
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(image_array.astype(float), sigma=blur_strength)
    
    # Create binary mask - pixels darker than threshold
    mask = (blurred < threshold).astype(np.uint8) * 255
    
    # Remove noise with morphological operations
    from scipy.ndimage import binary_erosion, binary_dilation
    mask_binary = mask > 0
    mask_binary = binary_erosion(mask_binary, iterations=1)
    mask_binary = binary_dilation(mask_binary, iterations=2)
    
    return (mask_binary * 255).astype(np.uint8)

def extract_data_points(mask):
    """
    Extract all data points (pixel coordinates) from the curve mask
    Returns sorted list of (x, y) coordinates
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([])
    
    # Combine and sort by x coordinate for meaningful output
    points = np.column_stack((xs, ys))
    points = points[np.argsort(points[:, 0])]
    return points

def get_boundary_box(image_pil):
    """
    Let user define the graph area boundaries to focus extraction
    """
    return {
        'left': 50,
        'right': image_pil.size[0] - 50,
        'top': 50,
        'bottom': image_pil.size[1] - 50
    }

def create_excel_export(points, filename="graph_data.xlsx"):
    """
    Create Excel file with extracted data points
    """
    df = pd.DataFrame(points, columns=['Pixel_X', 'Pixel_Y'])
    
    # Create Excel file in memory
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data Points')
        
        # Format the worksheet
        workbook = writer.book
        worksheet = writer.sheets['Data Points']
        worksheet.column_dimensions['A'].width = 15
        worksheet.column_dimensions['B'].width = 15
    
    buffer.seek(0)
    return buffer

# ===============================================
# STREAMLIT UI
# ===============================================

st.write("🚧 **No OCR? No Problem!** We extract raw data points from your graph.")

uploaded = st.file_uploader(
    "📊 Upload Graph Image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    
    # Display uploaded image
    st.image(image, caption="Your Graph", use_container_width=True)
    
    # Settings
    st.subheader("⚙️ Detection Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        blur = st.slider(
            "Blur Strength (higher = smoother curves)",
            min_value=1,
            max_value=15,
            value=5,
            step=1
        )
    
    with col2:
        threshold = st.slider(
            "Pixel Threshold (lower = detect darker lines)",
            min_value=50,
            max_value=250,
            value=200,
            step=10
        )
    
    # Process button
    if st.button("♾️ Extract Data Points"):
        st.info("🔍 Processing image...")
        
        # Detect curve mask
        mask = detect_curve_points(image, blur_strength=blur, threshold=threshold)
        
        # Extract points
        points = extract_data_points(mask)
        
        if len(points) > 0:
            st.success(f"✅ Extracted **{len(points)}** data points!")
            
            # Show visualizations
            st.subheader("📊 Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Detected Mask**")
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                ax1.imshow(mask, cmap="gray")
                ax1.set_title("Extracted Curve Mask")
                ax1.set_xlabel("Pixel X")
                ax1.set_ylabel("Pixel Y")
                st.pyplot(fig1)
            
            with col2:
                st.write("**Data Points Scatter Plot**")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.scatter(points[:, 0], points[:, 1], s=2, alpha=0.6, color='red')
                ax2.set_xlabel("Pixel X")
                ax2.set_ylabel("Pixel Y")
                ax2.set_title("Extracted Data Points")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            
            # Data preview
            st.subheader("📋 Data Preview")
            df_preview = pd.DataFrame(points[:20], columns=['Pixel_X', 'Pixel_Y'])
            st.dataframe(df_preview)
            st.write(f"Showing first 20 of {len(points)} points")
            
            # Export to Excel
            st.subheader("💾 Export Data")
            
            excel_buffer = create_excel_export(points)
            
            st.download_button(
                label="💣 Download as Excel (.xlsx)",
                data=excel_buffer,
                file_name="graph_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download"
            )
            
            # CSV export option
            csv_data = pd.DataFrame(points, columns=['Pixel_X', 'Pixel_Y']).to_csv(index=False)
            st.download_button(
                label="📊 Download as CSV",
                data=csv_data,
                file_name="graph_data.csv",
                mime="text/csv",
                key="csv_download"
            )
            
    # Statistics
    st.subheader("Data Statistics")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Total Points", len(points))
        with stats_col2:
            st.metric("X Range", f"{points[:, 0].min():.0f} - {points[:, 0].max():.0f}")
        with stats_col3:
            st.metric("Y Range", f"{points[:, 1].min():.0f} - {points[:, 1].max():.0f}")
        with stats_col4:
            st.metric("Spread", f"{len(np.unique(points[:, 0]))} unique X values")
    
        else:
            st.error("No data points detected! Try adjusting the settings.")

    else:
    st.info("Upload a graph image to extract data points")
