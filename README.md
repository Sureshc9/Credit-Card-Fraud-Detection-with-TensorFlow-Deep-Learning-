# Credit-Card-Fraud-Detection-with-TensorFlow-Deep-Learning-
This project implements a deep learning model to detect fraudulent credit card transactions using the Kaggle Credit Card Fraud dataset. The model is developed using TensorFlow's Functional API, with a complete pipeline including preprocessing, normalization, class imbalance handling, training, and evaluation.

# Overview: 
Fraud detection is a critical application of machine learning where identifying rare, malicious activities is essential. This project addresses the binary classification problem (fraud or not fraud) using transaction features.


# Dataset:
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download

Rows: 284,807

Features: 28 anonymized numerical features + Amount, Time, and the target column Class (0 = Legit, 1 = Fraud)

# Model Architecture:
Built with the TensorFlow Functional API, the architecture includes:
  
  Feature-wise Normalization layers
  
  Hidden layers: 128 → 64 units with ReLU activation
  
  Dropout regularization to reduce overfitting
  
  Output layer with Sigmoid for binary classification


  # Pipeline Steps:

1. Data Preprocessing
    Log transformation on Amount
    Drop Time, normalize inputs
    Train/validation/test split (70/15/15 stratified)
2. Input Pipeline
    tf.data.Dataset used for batching and prefetching
    Efficient and scalable for large data
3. Model Training
    Optimizer: Adam
    Loss: Binary Crossentropy
    Metrics: AUC, Precision, Recall
    Early stopping to avoid overfitting
4. Evaluation
    ROC AUC Score
    Confusion Matrix
    Classification Report

# Results:
| Metric            | Score |
| ----------------- | ----- |
| ROC AUC Score     | 0.97  |
| Precision (Fraud) | 0.92  |
| Recall (Fraud)    | 0.77  |
| F1-score          | 0.84  |

# Confusion Matrix:


# Visualizations:
  Training metrics over epochs: loss, precision, recall, AUC
  Confusion matrix heatmap
  ROC and Precision-Recall curves

# Technologies Used:
  Python 3
  TensorFlow 2.x (Functional API)
  NumPy, Pandas
  Scikit-learn
  Seaborn & Matplotlib

# Future Improvements:
  SMOTE / class weighting (already partially addressed)
  Hyperparameter tuning
  XGBoost baseline comparison
  Deploy as a REST API or Streamlit dashboard

  
