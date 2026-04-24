"""
Professional Document Scanner - WITH BETTER ERROR HANDLING
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from datetime import datetime

# Import utility modules
from utils.document_detector import DocumentDetector
from utils.perspective_corrector import PerspectiveCorrector
from utils.image_enhancer import ImageEnhancer

# Page configuration
st.set_page_config(
    page_title="Professional Document Scanner",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-text {
        color: #00C853;
        font-weight: bold;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #FF9800;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📄 Professional Document Scanner</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    enhancement_mode = st.selectbox(
        "Enhancement Mode",
        ["color", "bw", "grayscale"],
        help="Color: Keep colors, BW: Black & white, Grayscale: Gray tones"
    )
    
    use_fallback = st.checkbox("Enable Fallback Mode", value=True, 
                               help="Use full image if document not detected")
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Upload a clear photo of a document
    2. System uses 4 detection strategies
    3. Applies perspective correction
    4. Download the scanned result
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - Use good, even lighting
    - Place document on contrasting background
    - Capture entire document in frame
    - Avoid shadows and glare
    - Keep document flat
    """)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Input Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )

with col2:
    st.markdown("### 📥 Scanned Output")

def preprocess_image(image):
    """Preprocess image to improve detection"""
    # Resize if too large
    height, width = image.shape[:2]
    if width > 1500:
        scale = 1500 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Enhance contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def process_image(image, mode, use_fallback):
    """Complete image processing pipeline"""
    start_time = time.time()
    
    try:
        # Preprocess image
        preprocessed = preprocess_image(image)
        
        # Step 1: Detect document
        corners, contour, edges = DocumentDetector.detect_document(preprocessed)
        
        if corners is None:
            if use_fallback:
                st.warning("⚠️ Document not detected. Using full image as fallback.")
                # Use full image as document
                height, width = preprocessed.shape[:2]
                corners = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
            else:
                return None, "Could not detect document. Please ensure the document is clearly visible."
        
        # Step 2: Apply perspective transform
        scanned = PerspectiveCorrector.apply_perspective_transform(preprocessed, corners)
        
        if scanned is None or scanned.size == 0:
            return None, "Perspective transform failed"
        
        # Step 3: Enhance image
        enhanced = ImageEnhancer.enhance_document(scanned, mode=mode)
        
        processing_time = time.time() - start_time
        
        return enhanced, f"Success! Document scanned in {processing_time:.2f} seconds"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

# Process uploaded image
if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        st.error("Could not read the image. Please try another file.")
    else:
        # Display original
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        col1.image(original_rgb, use_column_width=True)
        
        # Show processing status in output column
        with col2:
            with st.spinner("Processing document..."):
                # Create progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                result, status = process_image(image, enhancement_mode, use_fallback)
                progress_bar.empty()
        
        if result is not None:
            # Display result
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            col2.image(result_rgb, use_column_width=True)
            
            st.success(f"✅ {status}")
            
            # Download button
            result_pil = Image.fromarray(result_rgb)
            img_bytes = io.BytesIO()
            result_pil.save(img_bytes, format='JPEG', quality=95)
            img_bytes.seek(0)
            
            st.download_button(
                label="📥 Download Scanned Image",
                data=img_bytes,
                file_name=f"scanned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
            
        else:
            col2.error(f"❌ {status}")
            
            # Show detailed help
            st.markdown(f"""
            <div class="warning-box">
            <strong>🔍 Detection Failed - Help Center:</strong><br><br>
            
            <strong>Common Causes & Solutions:</strong><br>
            • 📷 <strong>Poor Lighting:</strong> Ensure the document is well-lit and evenly illuminated<br>
            • 🎨 <strong>Low Contrast:</strong> Place document on a contrasting background (white paper on dark surface)<br>
            • 📏 <strong>Document too small:</strong> Move camera closer to the document<br>
            • 🔄 <strong>Too much angle:</strong> Try to capture the document more directly<br>
            • 🌑 <strong>Shadows:</strong> Avoid shadows falling across the document<br>
            • 📱 <strong>Image quality:</strong> Ensure image is not blurry or out of focus<br><br>
            
            <strong>Try these tips:</strong><br>
            1. Place document on a flat, contrasting surface<br>
            2. Ensure good lighting (natural light works best)<br>
            3. Hold camera steady and parallel to document<br>
            4. Make sure entire document is visible in frame<br>
            5. Avoid reflections and glare<br>
            6. Enable "Fallback Mode" in sidebar if disabled<br>
            </div>
            """, unsafe_allow_html=True)

else:
    col2.info("👈 Upload an image to get started")
    
    # Feature showcase
    st.markdown("---")
    st.markdown("### 🚀 Features")
    
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    
    with col_info1:
        st.markdown("**🔍 4 Detection Strategies**")
        st.caption("Edge detection, Color segmentation, Morphology, Hough lines")
    
    with col_info2:
        st.markdown("**📐 Perspective Correction**")
        st.caption("Straightens tilted documents automatically")
    
    with col_info3:
        st.markdown("**✨ Image Enhancement**")
        st.caption("Improves brightness, contrast & sharpness")
    
    with col_info4:
        st.markdown("**📑 Fallback Mode**")
        st.caption("Works even when document detection fails")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Professional Document Scanner | 4 Detection Strategies | Built with OpenCV & Streamlit</p>",
    unsafe_allow_html=True
)