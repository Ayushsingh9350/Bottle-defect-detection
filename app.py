import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 128
THRESHOLD = 0.00094  # updated from your latest run

st.set_page_config(page_title="Bottle Defect Detector", layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('bottle_autoencoder.h5')

autoencoder = load_model()

st.title("🍾 Manufacturing Defect Detection")
st.write(
    "Upload an image of a bottle. The model was trained **only on normal (defect-free) bottles**, "
    "so it flags anomalies based on how well it can reconstruct the image — "
    "unfamiliar defects reconstruct poorly, producing a visible error signature."
)

uploaded_file = st.file_uploader("Upload a bottle image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    input_batch = np.expand_dims(img_array, axis=0)

    reconstruction = autoencoder.predict(input_batch)[0]
    error_map = np.mean(np.square(img_array - reconstruction), axis=-1)
    overall_error = np.mean(np.square(img_array - reconstruction))

    is_defective = overall_error >= THRESHOLD

    col1, col2 = st.columns([1, 2])
    with col1:
        if is_defective:
            st.error(f"⚠️ DEFECTIVE  \nReconstruction error: {overall_error:.5f}")
        else:
            st.success(f"✅ NORMAL  \nReconstruction error: {overall_error:.5f}")
        st.caption(f"Decision threshold: {THRESHOLD:.5f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_array); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].imshow(reconstruction); axes[1].set_title("Reconstructed"); axes[1].axis('off')
    axes[2].imshow(error_map, cmap='hot'); axes[2].set_title("Anomaly Heatmap"); axes[2].axis('off')
    st.pyplot(fig)

    st.caption(
        "The heatmap shows per-pixel reconstruction error — brighter regions indicate "
        "areas the model struggled to reconstruct, often corresponding to defect locations."
    )
else:
    st.info("👆 Upload a bottle image to get a prediction. Try test images from the MVTec AD 'test' folder.")