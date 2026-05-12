import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from PIL import Image

# Function to apply K-means clustering on the image
def apply_kmeans(image, num_clusters):
    pixel_values = image.reshape((-1, 3)).astype('float32')
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixel_values)
    labels = kmeans.labels_
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_image = centers[labels.flatten()]
    return segmented_image.reshape(image.shape)

# Function to apply GMM clustering on the image
def apply_gmm(image, num_clusters):
    pixel_values = image.reshape((-1, 3)).astype('float32')
    gmm = GaussianMixture(n_components=num_clusters, random_state=0).fit(pixel_values)
    labels = gmm.predict(pixel_values)
    centers = np.uint8(gmm.means_)
    segmented_image = centers[labels.flatten()]
    return segmented_image.reshape(image.shape)

# Streamlit UI
st.title('Image Segmentation using K-means and GMM')
st.sidebar.header('Settings')

# Input for number of clusters
num_clusters = st.sidebar.number_input('Enter Number of Clusters', min_value=2, max_value=30, value=3, step=1)

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    # Display the original image
    st.image(image, caption='Original Image', use_column_width=True)

    # Apply K-means and GMM to the image
    kmeans_image = apply_kmeans(image, num_clusters)
    gmm_image = apply_gmm(image, num_clusters)

    # Create two columns for side-by-side comparison
    col1, col2 = st.columns(2)
    with col1:
        st.image(kmeans_image, caption=f'K-means {num_clusters} Clusters', use_column_width=True)
    with col2:
        st.image(gmm_image, caption=f'GMM {num_clusters} Clusters', use_column_width=True)

# Footer
st.markdown('---')
st.markdown('*Developed by Dr. Jishan Ahmed*')

# To run this app:
# Save this script as a .py file and execute it with the Streamlit command in your environment.
