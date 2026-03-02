import streamlit as st
from PIL import Image
from utils import PlantDiseaseModel
import os

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
# Custom CSS (Premium Styling)
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
# Sidebar Section
# ==============================

st.sidebar.title("🌿 Project Overview")

st.sidebar.markdown("### 📌 What This Project Does")
st.sidebar.write("""
This AI system detects plant diseases from leaf images using
a fine-tuned EfficientNet deep learning model trained on
the PlantVillage dataset.
""")

st.sidebar.markdown("### 🔬 Model Details")
st.sidebar.write("""
- Architecture: EfficientNetB0  
- Transfer Learning + Fine-Tuning  
- Image Size: 224x224  
- Classes: 38  
- Test Accuracy: 98.04%  
- Macro F1 Score: 0.98  
""")

st.sidebar.markdown("### 📖 How To Use")
st.sidebar.write("""
1. Upload a clear leaf image  
2. Wait for AI analysis  
3. View prediction & confidence  

⚠ Best results with clear, well-lit images.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using TensorFlow & Streamlit")

# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():
    return PlantDiseaseModel(MODEL_PATH, CLASS_NAMES_PATH)

model = load_model()

# ==============================
# Main Header
# ==============================

st.markdown("<h1>🌱 Plant Disease Detection AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Upload a leaf image to detect plant disease instantly.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ==============================
# Upload Section
# ==============================

st.markdown("## 📤 Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Choose an image file",
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
            predicted_class, confidence = model.predict(image)

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)

        st.markdown("### 🧠 Prediction Result")
        st.success(predicted_class)

        st.markdown("### 📊 Confidence Score")
        st.progress(float(confidence))
        st.write(f"{confidence * 100:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div class='footer'>Industry-Level Deep Learning Project | 38-Class Multi-Disease Classification</div>",
    unsafe_allow_html=True
)