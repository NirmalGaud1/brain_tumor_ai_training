import os

# --- Set environment variable BEFORE importing TensorFlow / Keras ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
from PIL import Image
import requests
import streamlit as st
import tensorflow as tf

# --- Page Configuration ---
st.set_page_config(page_title="Brain Tumor Detector", layout="centered", page_icon="🧠")

# --- Configuration ---
MODEL_DIR = "model"
AVAILABLE_MODELS = [
    "mobilenetv2_final.h5",
    "mobilenetv2_best.h5"
]

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/NirmalGaud1/brain_tumor_ai_training/main/"


# --- Download Model Function ---
@st.cache_data(show_spinner=False)
def ensure_model(filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            header = f.read(8)
        if not header.startswith(b"\x89HDF"):
            os.remove(model_path)

    if not os.path.exists(model_path):
        url = GITHUB_RAW_BASE + filename
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, stream=True, headers=headers, timeout=120)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            if total_size < 1000:
                st.error(f"Downloaded file `{filename}` is too small ({total_size} bytes). Check repository URL.")
                st.stop()

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        except Exception as e:
            st.error(f"Failed to download {filename}: {e}")
            st.stop()

    with open(model_path, "rb") as f:
        header = f.read(8)
        if not header.startswith(b"\x89HDF"):
            st.error(f"❌ `{filename}` is not a valid .h5 file.")
            st.stop()

    return model_path


# --- Safe Keras Model Loader ---
@st.cache_resource
def load_keras_model(path):
    """
    Attempts to load model using native Keras load_model.
    If Keras 3 deserialization fails due to list-formatted axis in BatchNormalization,
    it reloads using tf_keras or weights reconstruction.
    """
    try:
        # Standard load attempt
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        # Fallback for Streamlit Cloud running Keras 3
        try:
            import tf_keras
            return tf_keras.models.load_model(path, compile=False)
        except ImportError:
            # Re-raise original exception if tf_keras is not installed
            raise


# --- Image Preprocessing ---
def preprocess_image(image, target_size=(128, 128)):
    resized_img = image.convert("RGB").resize(target_size)
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# --- Sidebar UI ---
st.sidebar.title("⚙️ Settings")
selected_model = st.sidebar.selectbox("Choose a Model (.h5)", AVAILABLE_MODELS)

# --- Model Loading ---
with st.spinner(f"Loading `{selected_model}`..."):
    model_path = ensure_model(selected_model)
    try:
        model = load_keras_model(model_path)
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        st.info(
            "💡 **Streamlit Cloud Deployment Tip:** Add `tf-keras` to your `requirements.txt` file "
            "to resolve Keras 2 -> Keras 3 deserialization issues."
        )
        st.stop()

# Determine expected shape
try:
    input_shape = model.input_shape[1:3]
    if input_shape[0] is None:
        input_shape = (128, 128)
except Exception:
    input_shape = (128, 128)

# --- Main Interface ---
st.title("🧠 Brain Tumor Detection")
st.write(f"Active Keras Model: **{selected_model}**")

uploaded_file = st.file_uploader("Upload an MRI scan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    with st.spinner("Analyzing scan..."):
        input_data = preprocess_image(image, target_size=input_shape)
        predictions = model.predict(input_data)
        prob = float(predictions[0][0])

    tumor_detected = prob > 0.5
    confidence = (prob if tumor_detected else 1.0 - prob) * 100.0

    st.markdown("---")
    st.subheader("Prediction Analysis")

    if tumor_detected:
        st.error("### 🧬 Tumor Detected")
    else:
        st.success("### ✅ No Tumor Detected")

    st.write(f"Confidence Level: **{confidence:.2f}%**")
    st.progress(min(max(prob, 0.0), 1.0))
