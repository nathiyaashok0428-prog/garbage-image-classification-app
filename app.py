# =====================================================
# Garbage Classification AI App (Streamlit)
# =====================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "models/transfer_model.keras"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="Garbage Classification AI",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Garbage Image Classification AI")
st.markdown(
    "Upload an image and the AI will classify the garbage type."
)

# -------------------------------------------------
# LOAD MODEL (Cached)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -------------------------------------------------
# IMAGE PREPROCESS FUNCTION
# -------------------------------------------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Garbage Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    # -------------------------
    # PREDICTION
    # -------------------------
    img_array = preprocess_image(image)

    predictions = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    with col2:
        st.subheader("Prediction Result")

        st.success(f"🧠 Predicted Class: **{predicted_class.upper()}**")
        st.info(f"🎯 Confidence: **{confidence:.2f}%**")

        # Probability chart
        st.subheader("Prediction Probabilities")

        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(CLASS_NAMES, predictions)
        ax.set_xlabel("Probability")
        ax.set_xlim(0,1)
        st.pyplot(fig)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("Deep Learning Project — Garbage Image Classification")
