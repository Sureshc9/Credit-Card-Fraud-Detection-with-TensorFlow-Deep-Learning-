# Credit Card Fraud Detection with TensorFlow (Deep Learning):

  This project implements a deep learning model using TensorFlow Functional API to detect fraudulent transactions from credit card data. The solution focuses on handling class imbalance, feature normalization, and real-world model evaluation metrics like AUC, Precision, and Recall.


# Overview:

Fraud detection is a high-impact area of data science involving highly imbalanced classification. This project targets binary classification to predict whether a transaction is fraudulent (1) or legitimate (0) based on anonymized features.

# Dataset:

  Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download

  Rows: 284,807 transactions

  Frauds: ~492 (~0.17% of total)

  Features:
    * 28 anonymized PCA-like features (V1 to V28)
    * Amount, Time (preprocessed)
    * Class: Target label (0 = Legit, 1 = Fraud)

# Model Architecture:

Built with TensorFlow 2.x Functional API:

    Input (normalized)
           ↓
    Dense(128) → BatchNorm → Dropout(0.5)
           ↓
    Dense(64) → Dropout(0.3)
           ↓
    Output Layer (Sigmoid)
    
  Activation: ReLU (hidden), Sigmoid (output)

  Loss: Binary Crossentropy
  
  Optimizer: Adam
  
  Metrics: AUC, Precision, Recall

# Pipeline Steps:

  | Step                   | Description                                              |
  | ---------------------- | -------------------------------------------------------- |
  | **1. Data Cleaning**   | Drop `Time`, log-transform `Amount`                      |
  | **2. Splitting**       | Train (70%) / Val (15%) / Test (15%) with stratification |
  | **3. Input Pipeline**  | Use `tf.data.Dataset` with prefetch and batching         |
  | **4. Feature Scaling** | Apply `tf.keras.layers.Normalization()` per feature      |
  | **5. Model Training**  | With `EarlyStopping` on validation AUC                   |
  | **6. Evaluation**      | Confusion Matrix, Classification Report, ROC AUC         |


# Evaluation Metrics:

  | Metric                | Value |
  | --------------------- | ----- |
  | **ROC AUC Score**     | 0.97  |
  | **Precision (Fraud)** | 0.92  |
  | **Recall (Fraud)**    | 0.77  |
  | **F1-Score (Fraud)**  | 0.84  |


# 🌟 Visualizations:

  Model Training

  Classification Report

              precision    recall  f1-score   support
         0       1.00      1.00      1.00     42648
         1       0.92      0.77      0.84        74



  Confusion matrix heatm
  ![Unknown](https://github.com/user-attachments/assets/0c72abfd-40a4-4776-b16d-d0afb97328a9)
ap
  

# Technologies Used:

  * Python 3

  * TensorFlow 2.x (Functional API)

  * Pandas, NumPy

  * Scikit-learn

  * Seaborn & Matplotlib

# Future Work:

  * Introduce SMOTE or more robust class-weighting

  * Hyperparameter tuning

  * Try advanced models like XGBoost or LightGBM

  * Deploy using FastAPI or Streamlit dashboard for real-time prediction

# Inspiration:

  Fraud detection is critical in the fintech world. The key challenge is extreme class imbalance. This project demonstrates how deep learning and scalable pipelines can improve fraud classification, balancing recall and precision.
