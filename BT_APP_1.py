#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st

# --- This MUST be the first Streamlit command ---
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

import numpy as np
from PIL import Image
import tensorflow as tf

# --- Load the smallest TFLite model (dynamic quant) ---
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model/mobilenetv2_dynamic_quant.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # (1, 128, 128, 3)
IMG_SIZE = input_shape[1:3]  # (128, 128)

# --- Preprocessing function ---
def preprocess_image(image):
    # Resize to model input size
    image = image.resize(IMG_SIZE)
    # Convert to numpy array and normalize to [0,1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Streamlit UI ---
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image and the model will classify it as **tumor** or **no tumor**.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    # Use use_column_width=True for older Streamlit versions, or omit width parameter
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_data = preprocess_image(image)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction (sigmoid output)
    prob = float(output_data[0][0])
    if prob > 0.5:
        label = "🧬 Tumor detected"
        confidence = prob * 100
    else:
        label = "✅ No tumor detected"
        confidence = (1 - prob) * 100

    # Display result
    st.subheader("Prediction Result")
    st.write(f"**{label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Optional: show probability bar
    st.progress(prob if prob > 0.5 else 1 - prob)

