🌱 Soil Image Classification — ANNAM.AI Orientation Challenges
This project is a consolidated effort built for two Kaggle challenges during the ANNAM.AI Orientation. It tackles soil image classification using supervised learning (Challenge 1) and unsupervised anomaly detection (Challenge 2). The models are built using PyTorch and utilize best practices in preprocessing, training, and evaluation.

🧩 Problem Statements
🔹 Challenge 1: Supervised Soil Type Classification
A fine-tuned ResNet34 model is trained on labeled soil images to classify soil types.

Approach: Transfer learning using pretrained ResNet34.

Objective: Accurately predict soil type classes based on image data.

Key Techniques: Data augmentation, softmax classifier, transfer learning.

🔹 Challenge 2: Unsupervised Anomaly Detection using Autoencoders
A Convolutional Autoencoder is trained only on normal soil samples (label = 1). At inference time, anomalous images (label = 0) are detected using reconstruction error.

Approach: Unsupervised learning using Autoencoder.

Objective: Detect anomalous soil images based on poor reconstruction.

Key Techniques: Reconstruction error, MSE thresholding.

🧠 Architectures
✅ ResNet34 (Challenge 1)
Transfer learning pipeline based on ImageNet-pretrained ResNet34. Final fully connected layers are fine-tuned for multi-class classification.



🔍 Convolutional Autoencoder (Challenge 2)
Symmetric convolutional-deconvolutional architecture that learns to reconstruct only normal samples.

mathematica
Copy
Edit
Input 
  ↓
Conv2D → ReLU → Conv2D → ReLU → Conv2D → ReLU 
  ↓
Deconv2D → ReLU → Deconv2D → ReLU → Deconv2D → Sigmoid
  ↓
Output (Reconstructed Image)


🗂️ Folder Structure
bash
Copy
Edit
.
├── docs/cards/
│   ├── architecure.png            # ResNet34 architecture diagram
│   ├── ml-metrics.json            # Model evaluation metrics (Challenge 1)
│   └── project_card.ipynb         # Summary notebook
│
├── notebook/
│   └── soil_classification_annam.ipynb      # ResNet34 training + inference
│   └── soil_autoencoder_anomaly.ipynb       # Autoencoder training + anomaly detection
│
├── submission.csv                 # Sample Kaggle submission format
├── requirements.txt              # Python dependencies
├── transcript.txt                # Summary of key learnings
├── README.md                     # Project documentation
🧪 Dataset Overview
Source: Provided via Kaggle competition portal.

Common Files
train_labels.csv: Contains image_id, label (1 = normal, 0 = anomaly or class label).

test_ids.csv: Contains test image filenames.

train/, test/: Image directories for training and evaluation.

Preprocessing (both challenges)
Resize to 128x128

Normalize to ImageNet standards or [0,1]

Convert to tensor

Batched using PyTorch DataLoader

⚙️ Setup Instructions
These steps work for both Challenge 1 and Challenge 2.

1. Clone the Repository
bash
Copy
Edit
git clone <repo-url>
cd <repo-folder>
2. Create and Activate a Virtual Environment (Optional but Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🚀 How to Use
Challenge 1: ResNet34 Classifier
bash
Copy
Edit
jupyter notebook notebook/soil_classification_annam.ipynb
Go through the notebook to:

Load labeled image data

Apply transforms and augmentations

Train and validate the model

Evaluate metrics and save predictions to submission.csv

Challenge 2: Autoencoder for Anomaly Detection
bash
Copy
Edit
jupyter notebook notebook/soil_autoencoder_anomaly.ipynb
Steps include:

Load only label = 1 images for training

Train autoencoder to reconstruct normal soil images

Use MSE-based reconstruction error for anomaly detection

Generate test predictions and store in submission.csv

📊 Evaluation Metrics
Challenge 1: Accuracy, Precision, Recall, F1-score, and Confusion Matrix (saved in ml-metrics.json)

Challenge 2: MSE threshold-based anomaly labeling, ROC-AUC (calculated in notebook)

📌 Key Learnings
Fine-tuning pretrained models on domain-specific datasets offers fast convergence and high accuracy.

Autoencoders are effective for visual anomaly detection where labeled anomalous data is scarce.

Image preprocessing and architectural choices significantly impact performance.
