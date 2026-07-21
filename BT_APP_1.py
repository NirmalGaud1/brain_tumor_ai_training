import streamlit as st
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import urllib.request

# --- Configuration ---
MODEL_DIR = "model"
MODEL_FILENAME = "mobilenetv2_dynamic_quant.tflite"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# 🔽 CHANGE THIS URL to your hosted file
MODEL_URL = "https://github.com/NirmalGaud1/brain_tumor_ai_training/blob/main/mobilenetv2_dynamic_quant.tflite"

# --- Ensure model exists ---
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        st.info(f"Downloading model...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()

ensure_model()

# --- Load model ---
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = tuple(input_details[0]['shape'][1:3])

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- UI ---
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to classify it as **tumor** or **no tumor**.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob = float(output_data[0][0])

    label = "🧬 Tumor detected" if prob > 0.5 else "✅ No tumor detected"
    confidence = (prob if prob > 0.5 else 1 - prob) * 100

    st.subheader("Prediction Result")
    st.write(f"**{label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
    st.progress(prob if prob > 0.5 else 1 - prob)
