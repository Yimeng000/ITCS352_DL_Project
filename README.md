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

## Environment
- Python 3.10+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- streamlit

---

## Install dependencies
pip install -r requirements.txt

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

## Full Pipeline (Reproduction Steps)

To reproduce the results, follow these steps:

### Step 1: Prepare dataset
Place the BelgiumTS dataset under:
data/

Then run:
```bash
python prepare_belgiumts.py

---

### Step 2: Train model
Example (Model 4 - best model):

python train.py --model resnet18 --aug --epochs 20

---

### Step 3: Evaluate model

python evaluate.py --model_path path/to/model.pth

---

### Step 4: Run demo

streamlit run app.py

---

## Model Development (All model training and results are in the Baseline)

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

--

## Reproducing reported results
The repository includes pre-trained models and outputs:

- Model weights (.pth files)
- Confusion matrices
- Classification reports
- Training curves

These results can be found under:

baseline/outputs_*

This allows reproduction without retraining.
