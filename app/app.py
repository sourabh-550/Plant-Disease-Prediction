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
.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 20px;
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
# Load Model
# ==============================

@st.cache_resource
def load_model():
    return PlantDiseaseModel(MODEL_PATH, CLASS_NAMES_PATH)

model = load_model()

# ==============================
# Header
# ==============================

st.markdown("<h1>🌿 Plant Disease Detection AI System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Deep Learning powered plant disease classification using EfficientNet.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ==============================
# About Project Section
# ==============================

st.markdown("## 📌 What This Project Does")

st.write("""
This system uses a fine-tuned **EfficientNetB0 deep learning model**
to classify plant leaf images into one of **38 disease categories**.

It can detect:
- Multiple crop types
- Healthy vs diseased leaves
- Early and advanced disease stages

The model was trained on the **PlantVillage dataset**
and optimized using transfer learning and fine-tuning.
""")

# ==============================
# Model Details Section
# ==============================

with st.expander("🔬 Model Details"):
    st.write("""
    - Architecture: EfficientNetB0 (Transfer Learning)
    - Training Strategy:
        - Frozen backbone training
        - Fine-tuning top layers
    - Image Size: 224x224
    - Classes: 38
    - Test Accuracy: 98.04%
    - Macro F1 Score: 0.98
    """)

# ==============================
# How To Use Section
# ==============================

with st.expander("📖 How To Use"):
    st.write("""
    1. Upload a clear image of a plant leaf.
    2. Ensure the leaf is visible and not blurry.
    3. Wait for the AI system to analyze the image.
    4. View predicted disease class and confidence score.
    
    ⚠️ Note: Best results are obtained with clear, well-lit images.
    """)

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

# ==============================
# Footer
# ==============================

st.markdown(
    "<div class='footer'>Built with TensorFlow • EfficientNet • Streamlit | Developed as an Industry-Level ML Project</div>",
    unsafe_allow_html=True
)