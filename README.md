# 🐦 The Avian Observer — Bird Identifier

A deep learning-powered web application that identifies one of 25 bird species from an uploaded image.

---


## ✨ Features

- Drag & drop image upload
- Real-time bird prediction
- Confidence score display
- Clean responsive UI (Tailwind CSS)
- Flask backend with REST API

---

## 🧠 Project Overview

- **Frontend:** Static `index.html` using Tailwind CSS + vanilla JavaScript
- **Backend:** Flask API (`/predict`) for inference
- **Model:** TensorFlow/Keras (MobileNetV2-based classifier)
- **Classes:** Stored in `classes.txt`

---

## 📁 Project Structure
```
├── app.py
├── index.html
├── train.py
├── predict.py
├── classes.txt
├── bird_model.h5 (excluded). - Kaggle - 
├── train/ (excluded)
├── valid/ (excluded)
```

## About the Dataset
``https://www.kaggle.com/datasets/ichhadhari/indian-birds`` <br>
The "Indian-Birds-Species-Image-Classification" consists of 25 bird species found in India, including Asian Green Bee-eater, Brown-Headed Barbet, Cattle Egret, Common Kingfisher, Common Myna, Common Rosefinch, Common Tailorbird, Coppersmith Barbet, Forest Wagtail, Gray Wagtail, Hoopoe, House Crow, Indian Grey Hornbill, Indian Peacock, Indian Pitta, Indian Roller, Jungle Babbler, Northern Lapwing, Red-Wattled Lapwing, Ruddy Shelduck, Rufous Treepie, Sarus Crane, White-Breasted Kingfisher, White-Breasted Waterhen, and White Wagtail.

The dataset contains a total of 37,000 images split into train and validation sets in an 80:20 ratio, with 30,000 images in the training set and 7,500 images in the validation set. Each species has 1,500 images in the dataset. This dataset can be used for image classification tasks and to develop machine learning models to recognize different species of birds found in India.
