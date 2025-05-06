

🔎 Overview

Fraud detection is a high-impact area of data science involving highly imbalanced classification. This project targets binary classification to predict whether a transaction is fraudulent (1) or legitimate (0) based on anonymized features.

📈 Dataset

Source: Kaggle Credit Card Fraud Dataset

Rows: 284,807

Features: V1-V28 (PCA components), Amount, Time

Target: Class — 0 (Non-fraud), 1 (Fraud)



🛠️ Pipeline Steps

1. Data Preprocessing

Log transformation on Amount

Dropped Time

Normalized all input features

Stratified train/val/test split (70/15/15)

2. Input Pipeline

Built using tf.data.Dataset

Efficient batching and prefetching for scalability

3. Model Architecture

Input: Feature-wise normalization layers

Hidden Layers: 128 ➔ 64 neurons with ReLU activation

Dropout: 0.5 & 0.3 to prevent overfitting

Output: Sigmoid activation for binary classification

4. Model Training

Optimizer: Adam (lr=1e-3)

Loss: BinaryCrossentropy

Metrics: AUC, Precision, Recall

Early stopping: Monitors val_auc

5. Evaluation

ROC AUC

Confusion matrix

Classification report

🔬 Performance Metrics

Metric

Score

ROC AUC

0.9717

Precision (Fraud)

0.92

Recall (Fraud)

0.77

F1-Score (Fraud)

0.84



🌟 Visualizations

ROC & Precision-Recall Curves

Metric tracking over epochs (Loss, Precision, Recall, AUC)

Confusion matrix heatmap

📊 Technologies Used

Python 3

TensorFlow 2.x (Functional API)

Pandas, NumPy

Scikit-learn

Seaborn & Matplotlib

💡 Future Work

Introduce SMOTE or more robust class-weighting

Hyperparameter tuning

Try advanced models like XGBoost or LightGBM

Deploy using FastAPI or Streamlit dashboard for real-time prediction
