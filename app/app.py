import streamlit as st
from PIL import Image
from utils import PlantDiseaseModel
import os
import time
import io

# ==============================
# Configuration
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "plant_functional.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "..", "models", "class_names.json")

st.set_page_config(
    page_title="Plant Disease Detection AI",
    page_icon="🌱",
    layout="wide"
)

# ==============================
# Custom Styling
# ==============================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    text-align: center;
    font-weight: 800;
    background: -webkit-linear-gradient(#4CAF50, #00C853);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.prediction-card {
    background-color: #161b22;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #30363d;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 0.85em;
    padding-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Model (Cached)
# ==============================
@st.cache_resource
def load_model():
    return PlantDiseaseModel(MODEL_PATH, CLASS_NAMES_PATH)

model = load_model()

# ==============================
# Header Section
# ==============================
st.markdown("<h1>🌿 Plant Disease Detection using Deep Learning</h1>", unsafe_allow_html=True)

st.markdown("""
### 📌 Project Overview

This project uses **Transfer Learning with EfficientNetB0** to detect plant leaf diseases
across **38 different classes** from the PlantVillage dataset.

The model was fine-tuned and achieved:

- **98.04% Test Accuracy**
- **0.98 Macro F1 Score**
- **0.98 Weighted F1 Score**

It is deployed using **Streamlit Cloud** for real-time inference.
""")

st.markdown("---")

# ==============================
# Sidebar – Model Info
# ==============================
with st.sidebar:
    st.header("🔍 Model Information")
    st.write("**Architecture:** EfficientNetB0 (Fine-tuned)")
    st.write("**Input Size:** 224 x 224 x 3")
    st.write("**Classes:** 38")
    st.write("**Framework:** TensorFlow 2.20")
    st.write("**Deployment:** Streamlit Cloud")

    with st.expander("📊 View Model Summary"):
        summary_str = io.StringIO()
        model.model.summary(print_fn=lambda x: summary_str.write(x + "\n"))
        st.text(summary_str.getvalue())

# ==============================
# Image Upload
# ==============================
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# Prediction Section
# ==============================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing image with AI model..."):
            start_time = time.time()
            predicted_class, confidence = model.predict(image)
            inference_time = time.time() - start_time

        # Format class name
        formatted_label = predicted_class.replace("___", " - ").replace("_", " ")

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)

        st.markdown("### 🧠 Prediction")
        st.success(formatted_label)

        st.markdown("### 📊 Confidence Score")
        st.progress(float(confidence))
        st.write(f"**{confidence * 100:.2f}%**")

        if confidence < 0.70:
            st.warning("⚠️ Low confidence prediction. Try a clearer leaf image.")

        st.markdown("### ⏱ Inference Time")
        st.write(f"{inference_time:.3f} seconds")

        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown(
    "<div class='footer'>Built with TensorFlow, EfficientNetB0 & Streamlit | © 2026 Plant Disease Detection Project</div>",
    unsafe_allow_html=True
)