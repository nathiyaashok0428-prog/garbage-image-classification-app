import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CLASS_NAMES = ['cardboard','glass','metal','paper','plastic','trash']

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/final_industry_model.keras",
        compile=False
    )

model = load_model()

# ======================
# PREPROCESS IMAGE
# ======================

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image).astype("float32")

    # ⭐ EfficientNet preprocessing (VERY IMPORTANT)
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ======================
# UI
# ======================
st.title("♻️ Garbage Image Classification AI")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=350)

    img = preprocess_image(image)

    preds = model.predict(img, verbose=0)[0]

    pred_idx = np.argmax(preds)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = preds[pred_idx]

    st.success(f"Prediction: {pred_class.upper()} ({confidence*100:.2f}%)")

    if confidence < 0.55:
        st.warning("⚠ Low confidence — image may be unclear or too zoomed.")

    # ======================
    # PROBABILITY GRAPH
    # ======================
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(CLASS_NAMES, preds)
    ax.set_xlabel("Confidence")
    st.pyplot(fig)