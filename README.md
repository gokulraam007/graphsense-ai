# graphsense-ai

## Physics-aware Graph Digitizer with OCR, IV curve fitting, and Streamlit deployment

### 📋 Overview
GraphSense-AI is a Streamlit-based application that automatically extracts data from scientific graphs (IV curves, transmission spectra, etc.) using Optical Character Recognition (OCR) and physics-aware curve fitting models.

### 🎯 Features
- **✓ OCR-based Label Extraction** - Automatic detection of axis labels using PaddleOCR
- **✓ Physics-aware IV Curve Fitting** - Fits Shockley diode equation: I = Isc*(1-V/Voc)
- **✓ Image Processing Pipeline** - Binary mask generation and curve pixel extraction using scipy.ndimage
- **✓ Quality Validation** - Automated checks for monotonic current and positive power output
- **✓ Interactive Visualizations** - Extracted masks and fitted curves with matplotlib
- **✓ No cv2 Dependency** - Pure scipy/PIL implementation for Streamlit Cloud compatibility

### 🔧 Tech Stack
- **Frontend**: Streamlit
- **OCR**: PaddleOCR
- **Image Processing**: PIL (Pillow) + scipy.ndimage
- **Curve Fitting**: scipy.optimize.curve_fit
- **Visualization**: matplotlib
- **Deployment**: Streamlit Cloud

### 📦 Installation & Deployment

#### Local Setup
```bash
git clone https://github.com/gokulraam007/graphsense-ai.git
cd graphsense-ai
pip install -r requirements.txt
streamlit run streamlit_app.py
```

#### Streamlit Cloud Deployment
1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create new app pointing to this repository
4. Select `streamlit_app.py` as main file
5. Deploy!

**Live App**: https://graphsense-ai-pf8sfd5m2btkzgwwfnf7pd.streamlit.app/

### 🐛 Known Issues & Fixes

#### Issue 1: PaddleOCR API Error
**Error**: `PaddleOCR.predict() got an unexpected keyword argument 'cls'`

**Root Cause**: The old code was passing `cls=True` parameter directly to `ocr.ocr()` method. The PaddleOCR API uses `use_angle_cls=True` during initialization, not during prediction.

**Solution** (v3.0):
```python
# WRONG:
result = ocr.ocr(image_array, cls=True)  # ❌ This fails!

# CORRECT:
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Set during init
result = ocr.ocr(image_array)  # No cls parameter needed
```

#### Issue 2: cv2 Import Failures on Streamlit Cloud
**Error**: `ImportError: libGL.so.1 cannot be found`

**Root Cause**: OpenCV (cv2) has GUI dependencies that aren't available in Streamlit Cloud's headless environment.

**Solution** (v2.0):
- Removed cv2 completely
- Replaced `cv2.connectedComponentsWithStats()` with `scipy.ndimage.label()`
- Used PIL for image processing instead of cv2
- Added `packages.txt` with system dependencies

### ⚙️ Key Functions

#### `extract_numeric_labels_with_positions(image_pil, axis="x")`
Extracts numeric labels from graph axes using PaddleOCR
- **Input**: PIL Image
- **Output**: Sorted list of (pixel_position, value) tuples
- **Error Handling**: Try-except with detailed error messages

#### `fit_iv_curve(V, I)`
Fits physics-aware IV equation to extracted data
- **Model**: I = Isc * (1 - V/Voc)
- **Method**: scipy.optimize.curve_fit
- **Safety**: Returns zero function if fitting fails

#### `validate_iv_curve(V, I)`
Validates extracted curve metrics
- Checks: Monotonic current, positive power output
- Returns: Metrics dict and validation checks dict

### 📊 Workflow

```
1. Upload Image
   ↓
2. Convert to Binary Mask (PIL)
   ↓
3. Extract OCR Labels (PaddleOCR)
   ↓
4. Calibrate Axes (polyfit)
   ↓
5. Extract Curve Pixels (scipy.ndimage)
   ↓
6. Fit Physics Model (scipy.optimize)
   ↓
7. Validate Results
   ↓
8. Display Visualizations & Report
```

### 📝 Requirements
- Python 3.8+
- streamlit
- numpy
- opencv-python-headless (for system compatibility)
- matplotlib
- pillow
- scipy
- paddleocr
- paddlepaddle>=2.4.0

### 🚀 Future Enhancements
- [ ] YOLOv8 automatic curve detection
- [ ] Batch processing mode
- [ ] CSV/Excel export
- [ ] Uncertainty quantification (error bars)
- [ ] Support for multiple curve types (transmission, fluorescence, etc.)
- [ ] IEC-compliant report generation

### 📖 Documentation

#### Version History
- **v3.0** (Current): PaddleOCR API fix, robust error handling
- **v2.0**: Removed cv2 dependency, scipy.ndimage integration
- **v1.0**: Initial release with cv2

#### Debugging Tips
1. **No labels detected**: Ensure image has clear, readable axis labels
2. **Poor curve fit**: Check image quality and contrast
3. **OCR errors**: Try converting image to different format (PNG vs JPG)
4. **Slow processing**: First run downloads PaddleOCR models (~200MB)

### 🤝 Contributing
Feel free to submit issues and enhancement requests!

### 📄 License
MIT License

### ✍️ Author
**Gokul Raam**
- LinkedIn: [gokulraam007](https://linkedin.com/in/gokulraam007)
- GitHub: [gokulraam007](https://github.com/gokulraam007)

---

**Last Updated**: January 3, 2026
**Status**: ✅ Production Ready
