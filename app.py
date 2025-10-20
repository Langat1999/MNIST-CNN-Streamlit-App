import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from typing import Tuple

# -----------------------------
# 🧠 APP CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🧮",
    layout="centered"
)

st.title("🧠 MNIST Digit Classifier")
st.markdown("Upload an image of a handwritten digit (28×28 pixels) to classify it using a trained CNN model.")

# -----------------------------
# 🚀 MODEL LOADING (CACHED)
# -----------------------------
@st.cache_resource
def load_cnn_model():
    """Load the pre-trained MNIST CNN model once (cached)."""
    try:
        model = load_model("mnist_cnn_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_cnn_model()

def pad_and_resize_to_28(image: Image.Image) -> Image.Image:
    """Keep aspect ratio: pad to square and resize to 28x28."""
    image = image.convert("L")
    # invert if background is white (MNIST is white digit on black)
    if np.mean(image) > 127:
        image = ImageOps.invert(image)
    width, height = image.size
    max_side = max(width, height)
    padded = Image.new("L", (max_side, max_side), color=0)
    offset = ((max_side - width) // 2, (max_side - height) // 2)
    padded.paste(image, offset)
    return padded.resize((28, 28), Image.BILINEAR)

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Normalize to [0,1], add channel dim, shape (1,28,28,1)."""
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

# -----------------------------
# 🖼️ IMAGE UPLOAD & PREPROCESSING
# -----------------------------
uploaded_file = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        raw_image = Image.open(uploaded_file)
        processed = pad_and_resize_to_28(raw_image)
        st.image(processed, caption="Preprocessed Digit (28×28)", width=150)
        img_array = preprocess_image(processed)
    except Exception as e:
        st.error(f"Failed to read/process image: {e}")
        img_array = None

    # -----------------------------
    # 🔮 PREDICTION & OUTPUT
    # -----------------------------
    if model and img_array is not None:
        with st.spinner("Predicting..."):
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions)
            confidence = float(np.max(predictions)) * 100

        st.success(f"**Predicted Digit:** {predicted_class}")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

        # Display class probabilities
        st.subheader("🔢 Class Probabilities (Top-10)")
        probs = predictions[0]
        sorted_indices = np.argsort(probs)[::-1]
        sorted_labels = [str(i) for i in sorted_indices]
        sorted_values = [float(probs[i]) for i in sorted_indices]
        st.bar_chart({label: val for label, val in zip(sorted_labels, sorted_values)})
    else:
        st.error("Model not loaded. Please ensure 'mnist_cnn_model.h5' is in the same directory.")

else:
    st.info("👆 Upload a 28×28 pixel handwritten digit to begin.")
