import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 1. Load & Preprocess
# Pastikan file 'Raisin.csv' berada di folder yang sama dengan script ini
df = pd.read_csv('Raisin.csv')
X = df.drop('Class', axis=1)
y = df['Class']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 2. Membangun Arsitektur Deep Neural Network (DNN)
model = models.Sequential([
    layers.Dense(89, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # Hidden Layer 2
    layers.Dense(55, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    # Output Layer (Sigmoid untuk biner: Besni & Kecimen)
    layers.Dense(1, activation='sigmoid')
])

# 3. Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Training dengan Early Stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

print("Memulai Pelatihan DNN...")
history = model.fit(
    X_train, y_train,
    epochs=36,
    batch_size=32,
    validation_split=0.2, 
    callbacks=[early_stopping],
    verbose=1
)

# 5. Evaluasi
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred) * 100

print("\n" + "~" * 40)
print(f" >\\\< Akurasi Model DNN: {accuracy:.2f}% >\\\< ")
print("~" * 40 + "\n")

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- 6. VISUALISASI HASIL ---
# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix: Prediksi vs Aktual')
plt.xlabel('Prediksi (Predicted)')
plt.ylabel('Kenyataan (Actual)')
plt.show()