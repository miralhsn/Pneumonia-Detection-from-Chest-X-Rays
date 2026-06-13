import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

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
# BUILD MODEL (must match training architecture EXACTLY)
# -------------------------
def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model


# -------------------------
# LOAD MODEL WEIGHTS
# -------------------------
@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("model.weights.h5")
    return model

model = load_model()

# -------------------------
# IMAGE PREPROCESSING
# -------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)

    # grayscale → RGB fix
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "AI model for pneumonia detection using MobileNetV2 + transfer learning. "
    "For educational use only."
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
    confidence = float(prediction)

    # -------------------------
    # RESULT
    # -------------------------
    st.subheader("🧠 Prediction Result")

    if confidence > 0.5:
        st.error("⚠️ Pneumonia Detected")
        st.write("Model indicates signs of infection in lungs.")
    else:
        st.success("✅ Normal")
        st.write("No pneumonia detected.")

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

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("Built with Streamlit + TensorFlow 🧠 | Miral Hasan")
