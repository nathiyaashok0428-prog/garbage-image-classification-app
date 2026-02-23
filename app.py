import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("models/final_model.keras")
class_names = np.load("models/class_names.npy")

st.set_page_config(page_title="Garbage Classifier", layout="wide")

st.title("♻ Garbage Image Classification AI")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # =========================
    # PREPROCESS IMAGE (CRITICAL FIX)
    # =========================
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PREDICT
    # =========================
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.success(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")

    # =========================
    # PROBABILITY GRAPH
    # =========================
    fig, ax = plt.subplots()
    ax.barh(class_names, predictions)
    ax.set_xlabel("Confidence")
    st.pyplot(fig)