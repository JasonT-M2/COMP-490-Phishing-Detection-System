import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc, precision_recall_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------
# Paths / constants
# ---------------------------
CSV_PATH = "C:/onedrive/Desktop/Senior Proejct/Email3/Phishing_Email.csv"
LABEL_COL = "Email Type"        # target column
TEXT_COL = "Email Text"         # email content

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(CSV_PATH)

# Clean column names
df.columns = df.columns.str.strip()

# Drop unnecessary index column if exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Drop missing labels
df = df.dropna(subset=[LABEL_COL])

# ---------------------------
# Encode labels
# ---------------------------
le = LabelEncoder()
y = le.fit_transform(df[LABEL_COL])  # Safe Email -> 0, Phishing Email -> 1

# ---------------------------
# Convert text to TF-IDF features
# ---------------------------
vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(df[TEXT_COL].fillna('')).toarray()

# ---------------------------
# If numeric columns exist, include them
# ---------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.drop(LABEL_COL, errors='ignore').tolist()
if numeric_cols:
    X_numeric = df[numeric_cols].to_numpy()
    X_final = np.hstack([X_numeric, X_text])
else:
    X_final = X_text

# ---------------------------
# Scale features
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# ---------------------------
# Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)

print("Label distribution (train):")
print(pd.Series(y_train).value_counts())
print("Label distribution (test):")
print(pd.Series(y_test).value_counts())

# ---------------------------
# Compute class weights
# ---------------------------
unique_classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
class_weights = {cls: w for cls, w in zip(unique_classes, weights)}
print("Class Weights:", class_weights)

# ---------------------------
# Build model
# ---------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.summary()

# ---------------------------
# Train model
# ---------------------------
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train.astype(np.float32),
    validation_split=0.2,
    epochs=1,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[es],
    verbose=1
)
# Use raw string (r"...") or double backslashes to avoid escape issues
import os
# Use raw string r"" to avoid escape issues
model.save(r"C:\onedrive\Desktop\Senior Proejct\Project\phishing_model.keras")

import os

save_path = r"C:\onedrive\Desktop\Senior Proejct\Project\phishing_model.keras"
# Make sure folder exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

model.save(save_path)
print("Model saved successfully!")
# ---------------------------
# Evaluate
# ---------------------------
eval_res = model.evaluate(X_test, y_test.astype(np.float32), verbose=0)
print("\nTest results [loss, accuracy, auc]:", eval_res)

# ---------------------------
# Predictions & threshold
# ---------------------------
y_probs = model.predict(X_test).ravel()
threshold = 0.5
y_pred = (y_probs > threshold).astype(int)

# ---------------------------
# Labels for reports
# ---------------------------
classes = sorted(np.unique(y_test))
label_names = le.inverse_transform(classes)  # Converts 0/1 back to original strings

# ---------------------------
# Confusion matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (threshold={threshold})")
plt.show()

# ---------------------------
# Classification report
# ---------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=classes, target_names=label_names))

# ---------------------------
# ROC Curve
# ---------------------------
fpr, tpr, _ = roc_curve(y_test, y_probs, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ---------------------------
# Precision-Recall Curve
# ---------------------------
precision, recall, _ = precision_recall_curve(y_test, y_probs, pos_label=1)
plt.figure()
plt.plot(recall, precision, label="PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
