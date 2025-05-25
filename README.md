# 🌱 Soil Image Classification — ANNAM.AI Orientation Challenges

This project was developed during the **ANNAM.AI orientation Kaggle competition** and tackles two core challenges:

- **Challenge 1:** Soil type classification using a fine-tuned ResNet34 (supervised learning).
- **Challenge 2:** Anomaly detection in soil images using a convolutional autoencoder (unsupervised learning).

---

## 🧠 Architecture Overview

### 🔹 Challenge 1: ResNet34 Classifier

The model is built using a **transfer learning approach** with **ResNet34**, fine-tuned for soil classification. The pipeline involves:

- Image preprocessing using standard transforms.
- Training on labeled soil images.
- Evaluation and inference on test images.

![ResNet34 Architecture](./docs/cards/architecure.png)

---

### 🔹 Challenge 2: Autoencoder for Anomaly Detection

A **convolutional autoencoder** is trained only on normal soil images (`label = 1`). The reconstruction error is used to detect anomalies (`label = 0`) in test images.

![Autoencoder Architecture](./docs/cards/autoencoder.png)

---

## 📁 Folder Structure

```
.
├── docs/cards/
│   ├── architecure.png              # ResNet34 model architecture
│   ├── autoencoder.png              # Autoencoder architecture
│   ├── ml-metrics.json              # Evaluation metrics for Challenge 1
│   └── project_card.ipynb           # Combined summary notebook
│
├── notebook/
│   ├── soil_classification_annam.ipynb      # ResNet34 classification notebook
│   └── soil_autoencoder_anomaly.ipynb       # Autoencoder anomaly detection notebook
│
├── submission.csv                   # Sample Kaggle submission format
├── requirements.txt                 # Python dependencies
├── transcript.txt                   # Summary/transcript of project insights
├── README.md                        # Project documentation (you're reading it)
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🧪 Challenge 1 — ResNet34 Classifier

```bash
jupyter notebook notebook/soil_classification_annam.ipynb
```

Steps:
- Load and preprocess the soil type dataset.
- Fine-tune the ResNet34 model.
- Evaluate and visualize results.
- Generate predictions for submission.

---

### 🧪 Challenge 2 — Autoencoder Anomaly Detection

```bash
jupyter notebook notebook/soil_autoencoder_anomaly.ipynb
```

Steps:
- Use only images with `label = 1` to train the autoencoder.
- Reconstruct images and compute error.
- Flag anomalies based on reconstruction threshold.
- Generate predictions for submission.

---

### 📤 Submission

The final predictions from both challenges are stored in:

```
submission.csv
```

Ensure the format matches Kaggle requirements.

---

## 📊 Evaluation Metrics

- **Challenge 1 (ResNet34):** Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- **Challenge 2 (Autoencoder):** MSE reconstruction error, ROC-AUC (analyzed in notebook).

Detailed metrics are available in [`ml-metrics.json`](./docs/cards/ml-metrics.json) and visualized within the notebooks.

---

## 📜 Notes

- **Transfer learning** significantly boosts performance with fewer samples.
- The **autoencoder** generalizes well to unseen anomalies using only normal data.
- The project structure supports **scalability** and **modularity** for future soil-based tasks.

---

