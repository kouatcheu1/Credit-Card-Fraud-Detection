# Credit Card Fraud Detection Using Deep Learning & Anomaly Detection 💳🔍

## 📌 Introduction
This project implements a hybrid approach to credit card fraud detection using deep learning (Autoencoder) and traditional machine learning (Isolation Forest) methods. The system analyzes transaction patterns to identify potentially fraudulent activities, providing a robust comparison between different anomaly detection techniques.

## ❓ Problem Statement
Credit card fraud causes billions in losses annually. This project aims to:
- Detect fraudulent transactions using unsupervised learning techniques
- Compare the effectiveness of deep learning vs. traditional anomaly detection
- Provide a scalable solution for real-time fraud detection
- Minimize false positives while maintaining high fraud detection rates

## 📂 Dataset
The project uses the standard credit card fraud dataset containing transactions made by credit cards. The dataset includes:
- Time: Number of seconds elapsed between this transaction and the first transaction
- Amount: Transaction amount
- V1-V28: Principal components obtained with PCA transformation
- Class: 1 for fraudulent transactions, 0 for legitimate ones

## 🛠 Technologies Used
- **Python 3.x**
- **Libraries:**
  - TensorFlow/Keras - Deep learning implementation
  - scikit-learn - Machine learning algorithms
  - pandas - Data manipulation
  - numpy - Numerical operations
  - matplotlib - Data visualization
  - StandardScaler - Feature scaling

## 🔬 Methodology

### 1️⃣ Data Preprocessing
- Standard scaling of features
- Train-test split (80-20)
- Feature normalization

### 2️⃣ Model Implementation

#### Autoencoder Architecture
- Input Layer: Original feature dimensions
- Encoder Layers: 32 → 16 → 8 neurons
- Decoder Layers: 8 → 16 → 32 → Original dimensions
- Activation: ReLU (hidden layers), Sigmoid (output layer)
- Loss: Mean Squared Error
- Optimizer: Adam

#### Isolation Forest
- Estimators: 100
- Contamination: 0.01
- Random State: 42

### 3️⃣ Anomaly Detection Strategy
- **Autoencoder:** Uses reconstruction error with 95th percentile threshold
- **Isolation Forest:** Direct anomaly prediction with contamination factor

## ⚡ Model Comparison

| Model               | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | Accuracy |
|---------------------|------------------|---------------|-----------------|----------|
| **Autoencoder**     | **0.03**         | **0.90**      | **0.06**        | **0.95** |
| **Isolation Forest**| **0.10**         | **0.63**      | **0.18**        | **0.99** |

### 🔍 Key Insights:
- **Autoencoder** detects more fraud cases (**high recall: 90%**) but has a **low precision (3%)**, leading to more false positives.
- **Isolation Forest** is **more precise (10%)** but has a **lower recall (63%)**, meaning it misses more actual fraud cases.
- **Accuracy is higher for Isolation Forest (99%)**, but **fraud detection performance is better with Autoencoder**.

Results are visualized through:
- Training loss curves for Autoencoder
- Comparative bar plots for model metrics
- Detailed classification reports

## 🎯 Key Features and Functionality
- Dual model approach for robust fraud detection
- Automated threshold selection
- Comprehensive performance visualization
- Scalable preprocessing pipeline
- Model performance comparison

## 💡 Future Improvements
- Implement real-time prediction capabilities
- Add more anomaly detection algorithms
- Incorporate feature importance analysis
- Optimize model architectures
- Add cross-validation
- Implement ensemble methods

🚀 **Explore the Jupyter Notebook for complete code and insights!**
![Credit Card Fraud Detection](https://github.com/kouatcheu1/Credit-Card-Fraud-Detection/blob/main/Credit%20Card%20Fraud%20Detection.ipynb)
