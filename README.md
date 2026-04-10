🚦 Traffic Sign Classification – Baseline Model
📌 Project Overview

This project focuses on traffic sign classification using deep learning. The goal is to build a baseline model that can recognize traffic signs from images and serve as a foundation for further improvements.

Traffic sign recognition is an important task in real-world applications such as:

Autonomous driving
Driver assistance systems
Smart transportation

This baseline model provides a starting point for improving robustness under real-world conditions, such as:

Lighting variation
Different orientations
Noise and occlusion
📂 Project Structure<b>
<img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/006827ec-a9ca-49a1-a81a-dd6e33985d07" />


⚠️ Note: Dataset files are not included in this repository.

📊 Dataset

This project uses the Belgium Traffic Sign Dataset (BelgiumTSC).

Contains images of traffic signs from real-world road scenes
Multiple classes (traffic sign categories)
Variations in lighting, scale, and viewpoint
⚠️ Important

The dataset is not uploaded to GitHub due to size limitations.

You can download it from:
👉 https://btsd.ethz.ch/shareddata/

After downloading, place it into:

data/
⚙️ Baseline Model

The baseline model is implemented in:

baseline/model.py

Typical structure:

Convolutional layers (CNN)
Activation functions (ReLU)
Pooling layers
Fully connected layers for classification

This model is designed to:

Be simple and easy to train
Provide a reference for performance comparison
🏋️ Training

Training script:

baseline/train.py
Run training:
cd baseline
python train.py
Training pipeline:
Load dataset (dataset.py)
Preprocess images (resize, normalize)
Train CNN model
Save outputs (loss, accuracy, model weights)
📈 Outputs

Training results are saved in:

outputs_belgiumts_classification/

Includes:

Model weights (.pth)
Training logs
Accuracy / loss results
📌 Baseline Performance

Example results:

Test Accuracy: ~XX%
Test Loss: ~XX

These results serve as a baseline and will be improved in later models.
