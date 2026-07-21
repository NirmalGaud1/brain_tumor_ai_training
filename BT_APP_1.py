import streamlit as st
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

# --- Configuration ---
MODEL_DIR = "model"
MODEL_FILENAME = "mobilenetv2_dynamic_quant.tflite"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# ✅ CORRECT RAW URL (using raw.githubusercontent.com)
MODEL_URL = "https://raw.githubusercontent.com/NirmalGaud1/brain_tumor_ai_training/main/mobilenetv2_dynamic_quant.tflite"

# --- Ensure model file exists and is valid ---
def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Remove corrupt file if it exists but is invalid
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            header = f.read(4)
        if header != b'TFL3':
            os.remove(MODEL_PATH)
            st.warning("Removed corrupt model file. Re-downloading...")

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... This may take a moment.")
        try:
            # Use a proper user-agent to avoid GitHub blocking
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(MODEL_URL, stream=True, headers=headers, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if total_size < 1000:
                st.error(f"Downloaded file seems too small ({total_size} bytes). Check the URL.")
                st.stop()

            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

    # Final validation
    if os.path.getsize(MODEL_PATH) < 1000:
        st.error("Model file is too small or corrupt. Please check the download URL.")
        st.stop()

    with open(MODEL_PATH, 'rb') as f:
        header = f.read(4)
        if header != b'TFL3':
            st.error("File is not a valid TFLite model. Please check the download URL.")
            st.stop()

ensure_model()

# --- Load TFLite model ---
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = tuple(input_details[0]['shape'][1:3])  # (128, 128)

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
