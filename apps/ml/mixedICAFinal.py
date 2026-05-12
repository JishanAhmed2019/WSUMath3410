import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import soundfile as sf
import io
import cv2
from skimage import io as skio

def custom_styles():
    """Apply custom CSS styles for the Streamlit app."""
    st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
    }
    .signature {
        font-size:16px;
        font-weight:bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to perform ICA and reconstruct the image based on a given number of components
def reconstruct_image_ica(image, n_components):
    # Reshape the color image into 2D array where each row represents one color pixel
    reshaped_image = image.reshape(-1, 3)

    # Initialize FastICA with the given number of components
    ica = FastICA(n_components=n_components, random_state=0, whiten='unit-variance', max_iter=500)

    # Fit ICA on the color image data
    ica.fit(reshaped_image)

    # Transform the data to the independent components space and inverse transform it back
    transformed_ica = ica.transform(reshaped_image)
    restored_image = ica.inverse_transform(transformed_ica)

    # Reshape the data back to the original image shape
    restored_image = restored_image.reshape(image.shape)
    restored_image = np.clip(restored_image, 0, 255)  # Ensure valid pixel range

    return restored_image

# Existing functions for custom styles, audio and image loading, mixing, separation, and display...

def load_color_image_from_upload(file):
    """Load a color image from an uploaded file."""
    bytes_data = file.getvalue()
    image = skio.imread(io.BytesIO(bytes_data))
    return image

def display_reconstructed_images(image, reconstructed_images, n_components_list):
    """Display original and reconstructed images."""
    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        for n_components, img in zip(n_components_list, reconstructed_images):
            st.image(img.astype(np.uint8), caption=f"Reconstructed with {n_components} components", use_column_width=True)

def load_audio_from_upload(file, sr=22050, duration=30):
    """Load audio from an uploaded file."""
    bytes_data = file.getvalue()
    audio, _ = librosa.load(io.BytesIO(bytes_data), sr=sr, duration=duration)
    return audio

def mix_audios(audio1, audio2, weights):
    """Mix two audio signals with given weights."""
    return audio1 * weights[0] + audio2 * weights[1]

def separate_audio(mixed_signals, n_components=2):
    """Separate mixed audio signals using ICA."""
    ica = FastICA(n_components=n_components, random_state=0)
    separated_signals = ica.fit_transform(mixed_signals.T).T
    return separated_signals

def plot_signals(audios, titles, sr=22050):
    """Plot and display audio signals."""
    fig, axs = plt.subplots(len(audios), 1, figsize=(10, 2 * len(audios)))
    if len(audios) == 1:
        axs = [axs]  # Make iterable if only one plot
    times = np.linspace(0, len(audios[0])/sr, num=len(audios[0]))
    for i, (audio, title) in enumerate(zip(audios, titles)):
        axs[i].plot(times, audio)
        axs[i].set_title(title)
    plt.tight_layout()
    st.pyplot(fig)

def display_audio_button(audio, title, sr=22050):
    """Display an audio player with a button."""
    with st.expander(f"Click here to listen to {title}"):
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio, sr, format='wav')
        audio_buffer.seek(0)
        st.audio(audio_buffer, format='audio/wav')

def load_image_from_upload(file):
    """Load an image from an uploaded file."""
    bytes_data = file.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return image

def separate_images(image1, image2, n_components=2):
    """Separate mixed images using ICA."""
    S1 = image1
    S2 = cv2.resize(image2, (S1.shape[1], S1.shape[0]))
    w = np.array([[0.6, 0.4], [0.4, 0.6]])
    X1 = w[0, 0] * S1 + w[0, 1] * S2
    X2 = w[1, 0] * S1 + w[1, 1] * S2
    stacked_images = np.vstack((X1.flatten(), X2.flatten())).T
    ica = FastICA(n_components=n_components, max_iter=1000, tol=0.1)
    transformed_sources = ica.fit_transform(stacked_images).T
    separated_img1 = ((transformed_sources[0] - transformed_sources[0].min()) * (255 / (transformed_sources[0].max() - transformed_sources[0].min()))).astype(np.uint8).reshape(S1.shape)
    separated_img2 = ((transformed_sources[1] - transformed_sources[1].min()) * (255 / (transformed_sources[1].max() - transformed_sources[1].min()))).astype(np.uint8).reshape(S2.shape)
    return separated_img1, separated_img2, X1.astype(np.uint8), X2.astype(np.uint8)

def display_image(image, title):
    """Display an image with a title."""
    st.image(image, caption=title, use_column_width=True)

def main():
    custom_styles()  # Apply custom CSS styles
    st.image("math_horiz.png", use_column_width=True)
    st.title('Welcome to the Cocktail Party (18+)!')
    st.markdown("#### Choose the process: separate audio/images or reconstruct images using ICA.")

    process_choice = st.radio("", ('Audio Separation', 'Image Separation', 'Image Reconstruction'), horizontal=True)

    if process_choice == 'Audio Separation':
        st.header("🎸🎺🥁🎵 Audio Separation")
        audio1_file = st.file_uploader('Upload first audio file (.mp3, .wav)', type=['mp3', 'wav'], key="audio1")
        audio2_file = st.file_uploader('Upload second audio file (.mp3, .wav)', type=['mp3', 'wav'], key="audio2")
        
        if audio1_file and audio2_file:
            n_components_audio = st.number_input('Number of ICA components for audio', min_value=1, max_value=10, value=2, step=1, key="n_components_audio")
            
            if st.button("Process Audios"):
                audio1 = load_audio_from_upload(audio1_file)
                audio2 = load_audio_from_upload(audio2_file)
                mixed_audio1 = mix_audios(audio1, audio2, [0.6, 0.4])
                mixed_audio2 = mix_audios(audio1, audio2, [0.5, 0.5])
                mixed_signals = np.vstack([mixed_audio1, mixed_audio2])
                separated_audios = separate_audio(mixed_signals, n_components=n_components_audio)
                audios = [audio1, audio2, mixed_audio1, mixed_audio2, separated_audios[0], separated_audios[1]]
                titles = ['Original Audio 1', 'Original Audio 2', 'Mixed Audio 1', 'Mixed Audio 2', 'Separated Audio 1', 'Separated Audio 2']
                for audio, title in zip(audios, titles):
                    plot_signals([audio], [title])
                    display_audio_button(audio, title)

    elif process_choice == 'Image Separation':
        st.header("🌲🖼️🏯🏰 Image Separation")
        image1_file = st.file_uploader('Upload first image file', type=['jpg', 'png'], key="image1")
        image2_file = st.file_uploader('Upload second image file', type=['jpg', 'png'], key="image2")
        if image1_file and image2_file:
            n_components_image = st.number_input('Number of ICA components for images', min_value=1, max_value=10, value=2, step=1, key="n_components_image")
            if st.button("Process Images"):
                image1 = load_image_from_upload(image1_file)
                image2 = load_image_from_upload(image2_file)
                separated_img1, separated_img2, mixed_img1, mixed_img2 = separate_images(image1, image2, n_components=n_components_image)
                display_image(image1, "Original Image 1")
                display_image(image2, "Original Image 2")
                display_image(mixed_img1, "Mixed Image 1")
                display_image(mixed_img2, "Mixed Image 2")
                display_image(separated_img1, "Separated Image 1")
                display_image(separated_img2, "Separated Image 2")

    elif process_choice == 'Image Reconstruction':
        st.header("🎨 Image Reconstruction")
        image_file = st.file_uploader('Upload an image file', type=['jpg', 'png', 'jpeg'], key="reconstruct_image")
        if image_file:
            n_components_image = st.number_input('Number of ICA components for reconstruction', min_value=1, max_value=3, value=1, step=1, key="n_components_reconstruct")
            if st.button("Reconstruct Image"):
                color_image = load_color_image_from_upload(image_file)
                reconstructed_images = []
                n_components_list = [n_components_image]
                for n_components in n_components_list:
                    reconstructed_image = reconstruct_image_ica(color_image, n_components)
                    reconstructed_images.append(reconstructed_image)
                # Using columns correctly
                col1, col2 = st.columns(2)
                with col1:
                    st.image(color_image, caption="Original Image", use_column_width=True)
                with col2:
                    for img in reconstructed_images:
                        st.image(img.astype(np.uint8), caption=f"Reconstructed with {n_components_image} components", use_column_width=True)

    # Signature
    st.markdown('<p class="signature">Created by Dr. Jishan Ahmed</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
