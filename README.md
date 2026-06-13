<h1 align="center">🫁 Pneumonia Detection System</h1>

<h3 align="center">
Deep Learning-Based Chest X-Ray Classification using CNN & Transfer Learning
</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN%20%2B%20MobileNetV2-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Medical%20AI-X--Ray%20Classification-red?style=for-the-badge"/>
</p>

---

## 🧠 Overview

This project is a **medical AI system** that detects pneumonia from chest X-rays using deep learning.

It combines:
- 🧠 Custom CNN model
- ⚡ MobileNetV2 (Transfer Learning)
- 🔍 Grad-CAM explainability for medical transparency

---

## 🫁 Sample Predictions (Grad-CAM)

<p align="center">
  <img src="samples/person22_virus_54.jpeg" width="350"/>
  <img src="samples/p22.png" width="350"/>
</p>

> 🔴 Red areas show lung regions influencing the model decision

---

## 🏗️ Model Architectures

### 🔹 CNN Model
- 3 Convolution layers + MaxPooling
- Dense layers with dropout
- Binary classification (NORMAL / PNEUMONIA)

### 🔹 MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet
- Fine-tuned for chest X-rays
- Lightweight + high accuracy

---

## 📊 Performance Comparison

| Model | Accuracy | Recall (PNEUMONIA) | Notes |
|------|---------|---------------------|------|
| CNN | 82% | 99% | Slight overfitting |
| MobileNetV2 | 87% | 98% | Better generalization |

---

## 🔍 Explainability (Grad-CAM)

Grad-CAM highlights the regions of the X-ray that influenced predictions.

- 🟥 Red → High importance regions  
- 🟦 Blue → Low importance regions  

This ensures **medical interpretability**, not just prediction.

---

## ⚙️ Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Grad-CAM  

---

## 📊 Key Insights

- Transfer learning significantly improved generalization  
- CNN model overfitted on training data  
- Grad-CAM validated that model focuses on lung regions  
- Proper preprocessing was critical for stability  

---

## ⚠️ Challenges

- Class imbalance in dataset  
- Overfitting in custom CNN  
- Interpreting medical predictions  
- Fine-tuning MobileNetV2 for grayscale images  

---

## 🔮 Future Improvements

- Web deployment for real-time diagnosis  
- Multi-disease classification (COVID, TB, etc.)  
- Hospital integration system  
- Advanced explainability dashboards  

---

## 🧠 Why This Project Matters

This project demonstrates:

- Real-world medical AI application  
- Deep learning + transfer learning expertise  
- Model interpretability (critical in healthcare AI)  
- End-to-end ML pipeline development  

---

## 👨‍💻 Author

**Miral Hasan**

<p align="center">
  <a href="https://github.com/miralhsn">GitHub</a> •
  <a href="https://linkedin.com/in/miral-hasan-26353b249">LinkedIn</a>
</p>

---

<p align="center">
⭐ If you find this project useful, consider starring the repository!
</p>
