import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Garbage Image Classification AI", layout="wide")

CLASS_NAMES = ['cardboard','glass','metal','paper','plastic','trash']

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/final_model.keras")

model = load_model()

st.title("♻️ Garbage Image Classification AI")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

# ===============================
# PREPROCESS FUNCTION (CRITICAL FIX)
# ===============================
def preprocess_image(image):

    # convert to RGB
    image = image.convert("RGB")

    # resize exactly like training
    image = image.resize((224,224))

    img_array = np.array(image)

    # convert to float32
    img_array = img_array.astype(np.float32)

    # MobilenetV2 preprocessing
    img_array = preprocess_input(img_array)

    # batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ===============================
# PREDICTION
# ===============================
if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=450)

    input_img = preprocess_image(image)

    preds = model.predict(input_img)[0]

    pred_idx = np.argmax(preds)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = preds[pred_idx]*100

    with col2:

        st.success(f"Prediction: {pred_class.upper()} ({confidence:.2f}%)")

        # clean probability graph
        fig, ax = plt.subplots(figsize=(6,4))

        ax.barh(CLASS_NAMES, preds)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0,1)

        st.pyplot(fig)