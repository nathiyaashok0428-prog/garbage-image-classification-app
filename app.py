import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(layout="wide")

# ===== LOAD =====
model = tf.keras.models.load_model("models/transfer_model.keras")

with open("models/class_names.json","r") as f:
    class_names = json.load(f)

IMG_SIZE = (224,224)

st.title("♻ Garbage Image Classification AI")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1,col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image")

    # ===== PREPROCESS =====
    img = image.resize(IMG_SIZE)
    img = np.array(img).astype("float32")
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # ===== PREDICT =====
    preds = model.predict(img)[0]

    pred_index = np.argmax(preds)
    confidence = preds[pred_index]

    with col2:
        st.success(
            f"Prediction: {class_names[pred_index]} ({confidence*100:.2f}%)"
        )

    # ===== GRAPH (FIXED) =====
    fig, ax = plt.subplots()
    ax.bar(class_names, preds)
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Probabilities")
    plt.xticks(rotation=45)

    st.pyplot(fig)