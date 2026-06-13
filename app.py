import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🫁 Pneumonia Detection System")
st.markdown("Upload a chest X-ray and get instant AI-powered diagnosis with explainability.")

# -----------------------------
# LOAD MODEL (cache for speed)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image)

    if len(image.shape) == 2:  # grayscale fix
        image = np.stack((image,) * 3, axis=-1)

    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Settings")
show_confidence = st.sidebar.checkbox("Show Confidence Score", True)

# -----------------------------
# UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    processed = preprocess(image)
    prediction = model.predict(processed)[0][0]

    # -----------------------------
    # RESULT
    # -----------------------------
    st.subheader("🧠 Prediction Result")

    if prediction > 0.5:
        st.error("⚠️ Pneumonia Detected")
    else:
        st.success("✅ Normal")

    # -----------------------------
    # CONFIDENCE SCORE
    # -----------------------------
    if show_confidence:
        st.markdown("### 📊 Confidence Score")

        st.progress(float(prediction))

        st.write(f"Probability of Pneumonia: **{prediction:.2f}**")

    # -----------------------------
    # INFO BOX
    # -----------------------------
    st.info(
        "This model is for educational purposes only and not a clinical diagnostic tool."
    )
