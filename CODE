# STEP 1: Install TensorFlow if not already
!pip install -q tensorflow scikit-learn pandas matplotlib seaborn

# STEP 2: Full Pipeline Code
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv('/content/creditcard.csv')
df['Amount_log'] = np.log1p(df['Amount'])
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Prepare features and target
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

def make_dataset(X, y, batch_size=2048, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_train, y_train)
val_ds = make_dataset(X_val, y_val, shuffle=False)
test_ds = make_dataset(X_test, y_test, shuffle=False)

# Normalization layers per column
all_inputs, encoded_features = [], []
for col in X.columns:
    inp = tf.keras.Input(shape=(1,), name=col)
    norm = tf.keras.layers.Normalization()
    norm.adapt(X_train[col].values.reshape(-1, 1))
    all_inputs.append(inp)
    encoded_features.append(norm(inp))

x = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=all_inputs, outputs=output)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name='auc')])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True)
model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[early_stop], verbose=1)

# Prediction and Evaluation
y_pred_proba = model.predict(dict(X_test)).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)

print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
