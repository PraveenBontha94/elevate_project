#  Face Mask Detection with Live Alert System

A real-time face mask detection system using a Convolutional Neural Network (CNN) integrated with OpenCV. The model detects whether a person is wearing a mask or not via webcam, and raises a visual alert when no mask is detected.

---

##  Project Overview

This project aims to help monitor mask compliance in public or workplace environments using live video feed. It leverages computer vision and deep learning to:
- Detect faces using Haar Cascades
- Classify images into "Mask" or "No Mask"
- Display real-time predictions with alert messages

---

##  Tools & Technologies

- Python 3.12
- TensorFlow / Keras
- OpenCV
- NumPy
- Haar Cascade Classifier
- Jupyter Notebook (for training & testing)
- Flask (optional for web deployment)

---

##  Model Architecture

The CNN model consists of:

- **Input Layer**: 128x128 RGB image
- **3 Convolutional Blocks**: Conv2D → MaxPooling2D → BatchNormalization
- **Fully Connected Layer**: Dense(128, ReLU) + Dropout(0.5)
- **Output Layer**: Dense(2, Softmax) for binary classification

```python
Conv2D(32, (3,3)) → MaxPooling2D → BatchNorm  
Conv2D(64, (3,3)) → MaxPooling2D → BatchNorm  
Conv2D(128, (3,3)) → MaxPooling2D → BatchNorm  
Flatten → Dense(128) → Dropout → Dense(2, softmax)
