# 🧠 CIFAR-10 Deep Learning Image Classifier (CustomTkinter GUI)

This project is a **deep learning-based image classification system** built using **TensorFlow** and **CustomTkinter**.  
It allows users to upload an image and get **real-time predictions** and **probabilities** for 10 CIFAR-10 classes such as airplanes, cars, birds, cats, and more.  

The system combines machine learning with a user-friendly desktop interface — perfect for learning or deployment demos.

---

## 🚀 Features

- ✅ Train a CNN on the CIFAR-10 dataset  
- ✅ Save and load the trained model automatically  
- ✅ Upload any image and get prediction + confidence  
- ✅ CustomTkinter-based modern GUI  
- ✅ Modular architecture (train, utils, GUI separated)  
- ✅ Easy to deploy using `run.py`

---

## 🧱 Project Structure

📦 cifar10_image_classifier
│
├── models/
│ └── my_model.keras # Saved trained model
│
├── src/
│ ├── utils.py # Preprocessing (split, normalize, encode)
│ ├── train_model.py # CNN model training
│ ├── model_loader.py # Loads trained model
│ └── ui_app.py # GUI for image upload & prediction
│
├── run.py # Main entry point (launch GUI)
├── requirements.txt # Required dependencies
└── README.md # Documentation