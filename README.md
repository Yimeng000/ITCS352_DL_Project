# Traffic Sign Recognition System (Deep Learning Project)

## Overview
This project focuses on traffic sign classification using deep learning techniques. The goal is to build a robust and accurate system capable of recognizing traffic signs under real-world conditions such as variations in lighting, orientation, and noise.

The system is developed as part of a Computer Vision course project and demonstrates an end-to-end pipeline including data preprocessing, model training, evaluation, and deployment using a web interface.

---

## Objectives
- Develop a deep learning model for traffic sign classification
- Improve model performance through iterative experiments
- Analyze model behavior using metrics such as accuracy and macro F1-score
- Deploy the trained model using an interactive web application

---

## Dataset
The project uses a BelgiumTS traffic sign dataset  

### Key characteristics:
- Multiple traffic sign categories (fine-grained classification)
- Real-world variations (lighting, blur, angle)
- Class imbalance across categories

### Preprocessing steps:
- Cropping traffic sign regions
- Image resizing
- Normalization
- Data augmentation (rotation, flipping, etc.)

---

## Model Development

The project follows an **iterative model improvement strategy** rather than isolated experiments.

### Model 1: SimpleCNN (Baseline)
- Basic CNN architecture
- Fast training and interpretable results
- Provides a reference for further improvements

### Model 2: CNN + Data Augmentation
- Introduces augmentation to improve generalization
- Helps reduce overfitting but may introduce noise

### Model 3: ResNet18 (Transfer Learning)
- Pretrained deep architecture
- Better feature extraction
- Improved performance on complex classes

### Model 4: ResNet18 + Data Augmentation
- Further improvements through augmentation

---

## Evaluation Metrics
- **Accuracy**
- **Macro F1-score** (important for class imbalance)
- Confusion Matrix
- Classification Report

The results show that while overall accuracy is high, some visually similar traffic signs remain challenging, especially triangular warning signs.

---

## Key Challenges
- Class imbalance
- Similar visual patterns between classes
- Overfitting in deeper models
- Sensitivity to preprocessing and augmentation

---

## Web Application (Streamlit)
The trained model is deployed using a Streamlit web app.

### Features:
- Upload traffic sign images
- Real-time prediction
- User-friendly interface

### Run locally:
```bash
streamlit run app.py
