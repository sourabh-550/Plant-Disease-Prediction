import streamlit as st
from PIL import Image
from utils import PlantDiseaseModel
import os

# ==============================
# Configuration
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "plant_model_clean.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "..", "models", "class_names.json")

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# ==============================
# Custom CSS (Premium Styling)
# ==============================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    text-align: center;
    font-weight: 700;
    background: -webkit-linear-gradient(#4CAF50, #00C853);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    padding: 0.5em 1em;
}
.prediction-card {
    background-color: #161b22;
    padding: 20px;
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
# Load Model
# ==============================

@st.cache_resource
def load_model():
    return PlantDiseaseModel(MODEL_PATH, CLASS_NAMES_PATH)

model = load_model()

# ==============================
# Header
# ==============================

st.markdown("<h1>🌿 Plant Disease Detection AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Upload a leaf image and get instant AI-powered disease prediction.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ==============================
# Upload Section
# ==============================

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# Prediction
# ==============================

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing image with AI model..."):
            predicted_class, confidence = model.predict(image)

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)

        st.markdown("### 🧠 Prediction")
        st.success(predicted_class)

        st.markdown("### 📊 Confidence")
        st.progress(float(confidence))
        st.write(f"{confidence * 100:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Footer
# ==============================

st.markdown("---")
st.markdown(
    "<div class='footer'>Built with TensorFlow, EfficientNet & Streamlit | Model Accuracy: 98.04%</div>",
    unsafe_allow_html=True
)