<h1 align="center">🫁 Pneumonia Detection System</h1>

<h3 align="center">Deep Learning-Based Chest X-Ray Classification using CNN & MobileNetV2</h3>

<p align="center">
  <a href="https://YOUR-STREAMLIT-LINK.streamlit.app">
    🚀 Live Demo
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN%20%2B%20Transfer%20Learning-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-MobileNetV2-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Deployment-Streamlit-red?style=for-the-badge"/>
</p>

---

## 🧠 Overview

This project uses **deep learning models** to detect pneumonia from chest X-ray images.

It combines:
- Custom CNN model
- Transfer Learning using MobileNetV2
- Model interpretability using Grad-CAM

---

## 🚀 Live Demo

> Upload a chest X-ray and get instant prediction

<p align="center">
  <img src="assets/demo.gif" width="700"/>
</p>

---

## 🏗️ Model Architecture

### 🔹 CNN Model
- 3 Convolution layers
- MaxPooling layers
- Dense + Dropout
- Binary classification

### 🔹 MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet
- Custom classification head
- Fine-tuned for medical images

---

## 📊 Performance

| Model | Accuracy | Recall (PNEUMONIA) |
|------|---------|--------------------|
| CNN | 82% | 99% |
| MobileNetV2 | 87% | 98% |

---

## 🔍 Explainability (Grad-CAM)

Grad-CAM highlights regions in X-rays that influenced predictions.

- 🟥 Red → Important regions  
- 🟦 Blue → Less important regions  

---

## ⚙️ Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- Streamlit  
- Grad-CAM  

---

## 🧩 Challenges

- Handling class imbalance  
- Preventing overfitting in CNN  
- Improving generalization using transfer learning  
- Interpreting medical predictions  

---

## 🔮 Future Improvements

- Multi-class disease classification  
- Integration with hospital systems  
- Improved explainability dashboards  
- Deployment on edge devices  

---

## 🧠 Why This Project Matters

This project demonstrates:

- Real-world medical AI application  
- CNN + Transfer learning expertise  
- Model interpretability (Grad-CAM)  
- End-to-end deployment capability  

---

## 👨‍💻 Author

**Miral Hasan**

<p align="center">
  <a href="https://github.com/miralhsn">GitHub</a> •
  <a href="https://linkedin.com/in/miral-hasan-26353b249">LinkedIn</a>
</p>

---

<p align="center">
  ⭐ If you like this project, consider giving it a star!
</p>
