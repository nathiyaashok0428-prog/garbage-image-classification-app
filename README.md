# ♻️ Garbage Image Classification Using Deep Learning

## 📌 Overview

This project is an AI-powered garbage classification system that automatically classifies waste images into six categories using Deep Learning and Transfer Learning.

Classes:

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

The model is deployed using Streamlit for real-time image prediction.

---

## 🚀 Features

* Deep Learning Image Classification
* EfficientNetB0 Transfer Learning
* Data Augmentation
* Class Imbalance Handling
* Streamlit Web Application
* Prediction Confidence Graph

---

## 📂 Project Structure

```
garbage_image_classification/
│
├── data/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
├── models/
│   └── final_industry_model.keras
│
├── scripts/
│   └── 04_train_model.py
│
├── app.py
└── requirements.txt
```

---

## 🧠 Model Architecture

* EfficientNetB0 (Pretrained on ImageNet)
* Global Average Pooling
* Dropout Layer
* Dense Output Layer (6 classes)

---

## ⚙️ Training Strategy

### Stage 1 — Frozen Backbone

* Train classifier layer only.

### Stage 2 — Fine-Tuning

* Unfreeze backbone.
* Train with small learning rate.

---

## 📊 Data Augmentation

* Random Flip
* Random Rotation
* Random Zoom

Purpose:

* Improve generalization
* Reduce overfitting

---

## ▶️ How to Run Training

```
python scripts/04_train_model.py
```

Model will be saved as:

```
models/final_industry_model.keras
```

---

## 🌐 Run Streamlit App

```
streamlit run app.py
```

---

## 📈 Prediction Workflow

1. Upload image
2. Image preprocessing
3. Model prediction
4. Display class and confidence
5. Show probability graph

---

## 🧪 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Streamlit

---

## 🎯 Results

* Validation Accuracy: ~85%
* Real-time deployment successful

---

## 🔮 Future Scope

* Mobile deployment
* Real-time camera detection
* Larger dataset training
* Object detection integration

---

## 👨‍💻 Author

Nathiya Ashok
