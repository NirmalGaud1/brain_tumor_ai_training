import os
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
    "mobilenetv2_dynamic_quant.tflite",
    "mobilenetv2_float16_quant.tflite",
    "mobilenetv2_int8_quant.tflite",
]

# --- Correct URL for Git LFS media files ---
# Using media.githubusercontent.com allows fetching the true binary instead of the LFS text pointer
GITHUB_LFS_BASE = "https://media.githubusercontent.com/media/NirmalGaud1/brain_tumor_ai_training/main/"


# --- Function to Download and Validate Model ---
@st.cache_data(show_spinner=False)
def ensure_model(filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, filename)

    # Check existing file header for validity (TFLite models start with b'TFL3')
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            header = f.read(4)
        if header != b"TFL3":
            os.remove(model_path)

    # Download model if not present or removed due to corruption
    if not os.path.exists(model_path):
        url = GITHUB_LFS_BASE + filename
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, stream=True, headers=headers, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            if total_size < 1000:  # File too small, likely an error page or lfs pointer
                st.error(
                    f"Downloaded file `{filename}` is too small ({total_size} bytes). "
                    "Ensure the file exists at the source."
                )
                st.stop()

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        except Exception as e:
            st.error(f"Failed to download {filename}: {e}")
            st.stop()

    # Final verification check
    with open(model_path, "rb") as f:
        header = f.read(4)
        if header != b"TFL3":
            st.error(
                f"❌ File `{filename}` was downloaded, but it is not a valid TFLite binary. "
                "Verify Git LFS tracking on your GitHub repo."
            )
            st.stop()

    return model_path


# --- Model Loader ---
@st.cache_resource
def load_tflite_interpreter(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


# --- Image Preprocessing ---
def preprocess_image(image, input_shape):
    target_size = (input_shape[1], input_shape[2])  # (height, width)
    resized_img = image.convert("RGB").resize(target_size)
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# --- Sidebar UI ---
st.sidebar.title("⚙️ Settings")
selected_model = st.sidebar.selectbox("Choose a TFLite Model", AVAILABLE_MODELS)

# --- Model Loading Process ---
with st.spinner(f"Preparing `{selected_model}`..."):
    model_path = ensure_model(selected_model)
    try:
        interpreter = load_tflite_interpreter(model_path)
    except Exception as e:
        st.error(f"Error initializing TFLite interpreter: {e}")
        st.stop()

# Get model I/O details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]["shape"]

# --- Main Interface ---
st.title("🧠 Brain Tumor Detection")
st.write(f"Active Model: **{selected_model}**")

uploaded_file = st.file_uploader(
    "Upload an MRI scan...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Inference execution
    with st.spinner("Analyzing scan..."):
        input_data = preprocess_image(image, input_shape)

        # Quantitative input conversion if needed by quantized models
        if input_details[0]["dtype"] == np.int8 or input_details[0]["dtype"] == np.uint8:
            scale, zero_point = input_details[0]["quantization"]
            if scale > 0:
                input_data = (input_data / scale + zero_point).astype(input_details[0]["dtype"])

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])

        # Dequantize output if required
        if output_details[0]["dtype"] == np.int8 or output_details[0]["dtype"] == np.uint8:
            scale, zero_point = output_details[0]["quantization"]
            if scale > 0:
                output_data = (output_data.astype(np.float32) - zero_point) * scale

        prob = float(output_data[0][0])

    # Result formatting
    tumor_detected = prob > 0.5
    confidence = (prob if tumor_detected else 1.0 - prob) * 100.0

    st.markdown("---")
    st.subheader("Prediction Analysis")

    if tumor_detected:
        st.error(f"### 🧬 Tumor Detected")
    else:
        st.success(f"### ✅ No Tumor Detected")

    st.write(f"Confidence Level: **{confidence:.2f}%**")
    st.progress(min(max(prob, 0.0), 1.0))
