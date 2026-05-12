import streamlit as st
from PIL import Image, ImageFile
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# Page configuration and custom styling
st.set_page_config(
    page_title="SVD Image Compression",
    layout="wide",
    page_icon="🔢",
)

st.markdown("""
    <style>
    /* Hide Streamlit default menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Page Background */
    .main {
        background-color: #ffffff;
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background-color: #e3f2fd !important;
    }

    /* Titles */
    h1 {
        text-align: center;
        color: #003c8f;
        font-weight: bold;
    }
    
    /* Subheaders */
    h2, h3 {
        color: #004c8c;
    }

    /* Compression Stats */
    .compression-box {
        background-color: #f0f4f8;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: #333333;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #555555;
    }
    </style>
""", unsafe_allow_html=True)

# Allow large image processing
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# ------------------------------------------------------------------------
# Sidebar: Information and Instructions
st.sidebar.markdown("## 📌 About This App")
st.sidebar.write(
    "This app demonstrates **Singular Value Decomposition (SVD)** for image compression. "
    "Upload an image and adjust the rank slider to see how SVD reconstructs a compressed version."
)
st.sidebar.markdown("#### 📖 What is SVD?")
st.sidebar.write(
    "SVD factorizes a matrix into three components: \( U, Σ, V^T \). "
    "By using only a subset of singular values, we can approximate the original image while reducing storage."
)
st.sidebar.markdown("#### 🎯 How to Use?")
st.sidebar.write("1. Upload an image file (PNG, JPG, JPEG, BMP, or TIFF).")
st.sidebar.write("2. Adjust the slider to select the **rank for compression**.")
st.sidebar.write("3. View the **original and reconstructed images** side by side.")
st.sidebar.write("4. See **compression statistics and ratio**.")

# ------------------------------------------------------------------------
# App Title and Header
st.markdown("<h1>🔢 SVD-Based Image Compression</h1>", unsafe_allow_html=True)
st.write("Upload an image below and use the slider to see how different ranks affect image quality and compression.")

# ------------------------------------------------------------------------
# Image Upload
uploaded_file = st.file_uploader("📂 Upload an Image File", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file is None:
    st.info("👆 Please upload an image file to proceed.")
    st.stop()

# Load Image
try:
    image = Image.open(uploaded_file)
except Exception as e:
    st.error(f"❌ Error loading image: {e}")
    st.stop()

# Resize image if too large
max_dimensions = (1024, 1024)
if image.width > max_dimensions[0] or image.height > max_dimensions[1]:
    st.warning("⚠️ Resizing large image for processing efficiency.")
    try:
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        image.thumbnail(max_dimensions, resample_filter)
    except Exception as e:
        st.error(f"❌ Error resizing image: {e}")
        st.stop()

# Convert image to grayscale numpy array
image_np = np.array(image)
gray_image = np.array(image.convert("L"))

# ------------------------------------------------------------------------
# Compute image properties
image_height, image_width = gray_image.shape
num_pixels = image_height * image_width

# Determine max rank
max_rank = min(image_height, image_width)
default_rank = min(50, max_rank)

# ------------------------------------------------------------------------
# Layout: Original and Reconstructed Images
st.markdown("### 🖼️ Original vs. Reconstructed Image")

col_original, col_reconstructed = st.columns(2)

with col_original:
    st.subheader("📌 Original Image")
    plt.figure(figsize=(6, 6))
    plt.imshow(gray_image, cmap="gray")
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.clf()

with col_reconstructed:
    st.subheader("📌 Reconstructed Image")
    rank = st.slider("🎚️ Select Rank for Compression", min_value=1, max_value=max_rank, value=default_rank, step=1)

    with st.spinner("🔄 Processing SVD..."):
        # Perform SVD
        U, S, VT = np.linalg.svd(gray_image, full_matrices=False)
        S_diag = np.diag(S[:rank])  # Keep only first 'rank' singular values
        Xprox = U[:, :rank] @ S_diag @ VT[:rank, :]  # Reconstruct image
    
    plt.figure(figsize=(6, 6))
    plt.imshow(Xprox, cmap="gray")
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.clf()

# ------------------------------------------------------------------------
# Compute Compression Statistics
uncompressed_size = num_pixels  # Each pixel stored independently
compressed_size = rank * (image_width + image_height)  # Based on stored values in SVD
compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else float('inf')

# Compression Progress Bar
st.markdown("### 📊 Compression Statistics")
progress_bar = min(1.0, compression_ratio / 100)
st.progress(progress_bar)

# Display compression details
st.markdown(f"""
<div class='compression-box'>
    <p>📏 <strong>Image Size:</strong> {image_width} × {image_height} pixels</p>
    <p>🔢 <strong>Total Pixels:</strong> {num_pixels:,}</p>
    <p>📂 <strong>Uncompressed Size:</strong> {uncompressed_size:,} (proportional to total pixels)</p>
    <p>🗜️ <strong>Compressed Size (approx.):</strong> {compressed_size:,}</p>
    <p>⚡ <strong>Compression Ratio:</strong> {compression_ratio:.2f}</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------
# Footer
st.markdown("<p class='footer'>👨‍🏫 Developed by <strong>Dr. Jishan Ahmed</strong></p>", unsafe_allow_html=True)
