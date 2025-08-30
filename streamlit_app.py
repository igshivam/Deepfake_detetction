#run using streamlit run streamlit_app.py
# This script is a Streamlit application for detecting deepfake images using a pre-trained Xception model.

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import logging
from datetime import datetime

# App UI
st.set_page_config(page_title="DeepFake Detector", layout="centered")
st.title("🔍 DeepFake Detection App")
st.write("Upload a face image (299x299) to check if it's **Real** or **Fake**.")
# Setup logger
logging.basicConfig(filename='detection.log', level=logging.INFO)

def log_detection_result(filename, label, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"[{timestamp}] File: {filename} | Prediction: {label} | Confidence: {confidence:.2%}")

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("models/xception_deepfake_model.h5")

model = load_trained_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((299, 299)).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    # Prediction
    pred = model.predict(x)[0][0]
    label = "Fake" if pred > 0.5 else "Real"
    confidence = pred if pred > 0.5 else 1 - pred

    # Result
    st.markdown(f"### 🔍 Prediction: **{label}**")
    st.markdown(f"### 🔢 Confidence: `{confidence:.2%}`")
    st.markdown("### 📊 Model Architecture:"
                " [Xception](https://arxiv.org/abs/1610.02357)")
    