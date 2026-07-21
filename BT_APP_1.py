import streamlit as st
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

# --- Configuration ---
MODEL_DIR = "model"
AVAILABLE_MODELS = [
    "mobilenetv2_dynamic_quant.tflite",
    "mobilenetv2_float16_quant.tflite",
    "mobilenetv2_int8_quant.tflite",
    "mobilenetv2_final.h5",          # Keras model (not TFLite) – we'll ignore for TFLite
    "mobilenetv2_best.h5"            # Keras model
]

# Filter only TFLite models (ending with .tflite)
TFLITE_MODELS = [f for f in AVAILABLE_MODELS if f.endswith('.tflite')]

# --- GitHub raw URL base (modify if your repo structure differs) ---
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/NirmalGaud1/brain_tumor_ai_training/main/"

# --- Function to download and validate model ---
def ensure_model(filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, filename)

    # Remove invalid file if exists
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            header = f.read(4)
        if header != b'TFL3':
            os.remove(model_path)
            st.warning(f"Removed corrupt model file: {filename}")

    if not os.path.exists(model_path):
        url = GITHUB_RAW_BASE + filename
        st.info(f"Downloading {filename} ...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if total_size < 1000:
                st.error(f"File too small ({total_size} bytes). Check URL.")
                st.stop()

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            st.success(f"✅ {filename} downloaded successfully!")

        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

    # Validate again
    with open(model_path, 'rb') as f:
        header = f.read(4)
        if header != b'TFL3':
            st.error(f"❌ {filename} is not a valid TFLite model.")
            st.stop()

    return model_path

# --- Sidebar: model selection ---
st.sidebar.title("⚙️ Settings")
selected_model = st.sidebar.selectbox("Choose a model", TFLITE_MODELS)

# --- Load the selected model ---
model_path = ensure_model(selected_model)

@st.cache_resource
def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
IMG_SIZE = tuple(input_details[0]['shape'][1:3])  # (128, 128)

# --- Preprocessing ---
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Main UI ---
st.title("🧠 Brain Tumor Detection")
st.write(f"Using model: **{selected_model}**")

uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "jpeg", "png"])

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
