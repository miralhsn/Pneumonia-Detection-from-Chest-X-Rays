import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered"
)

# -------------------------
# TITLE
# -------------------------
st.title("🫁 Pneumonia Detection System")
st.markdown("Upload a chest X-ray image and get AI-based prediction with confidence score.")

# -------------------------
# LOAD MODEL (cached for speed)
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# -------------------------
# PREPROCESS FUNCTION
# -------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)

    # convert grayscale → RGB if needed
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------
# SIDEBAR INFO
# -------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This AI model detects pneumonia from chest X-rays using a MobileNetV2-based deep learning model. "
    "It is for educational purposes only."
)

# -------------------------
# UPLOAD IMAGE
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    # preprocess
    processed_img = preprocess_image(image)

    # prediction
    prediction = model.predict(processed_img)[0][0]

    # -------------------------
    # RESULT DISPLAY
    # -------------------------
    st.subheader("🧠 Prediction Result")

    confidence = float(prediction)

    if confidence > 0.5:
        st.error("⚠️ Pneumonia Detected")
        st.write("The model predicts signs of infection in the lungs.")
    else:
        st.success("✅ Normal")
        st.write("No visible signs of pneumonia detected.")

    # -------------------------
    # CONFIDENCE SCORE
    # -------------------------
    st.subheader("📊 Confidence Score")

    st.progress(confidence)
    st.write(f"Probability of Pneumonia: **{confidence:.2f}**")

    # interpretation
    if confidence > 0.8:
        st.warning("High confidence prediction")
    elif confidence > 0.5:
        st.info("Moderate confidence prediction")
    else:
        st.success("Low risk prediction")
