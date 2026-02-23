import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================= LOAD MODEL =================

model = tf.keras.models.load_model("models/transfer_model.keras")

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224,224)

st.title("♻ Garbage Image Classification")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # ===== PREPROCESS (🔥 FIXED) =====
    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)  # SAME AS TRAINING

    # ===== PREDICTION =====
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)

    st.subheader("Prediction")
    st.success(f"{class_names[class_idx]} ({confidence*100:.2f}%)")