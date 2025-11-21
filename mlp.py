import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

file_id = "Raisin.csv"
try:
    df = pd.read_csv(file_id)
except FileNotFoundError:
    print(f"Error: File {file_id} not found.")
    exit()

print("--- Data Awal ---")
print(df.head(min(901, len(df))))
print("\n")


X = df.drop('Class', axis=1) 
y = df['Class'] 

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Label Encoding Kelas:")
for i, name in enumerate(le.classes_):
    print(f"  {name}: {i}")
print("\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=2000,
    activation='relu',
    solver='adam',
    alpha=0.0001, 
    random_state=42,
    tol=1e-4,
    verbose=False 
)

print("Memulai Pelatihan Model MLP...")
mlp.fit(X_train, y_train)
print("Pelatihan Selesai.\n")

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100

print("-" * 40)
print(f" Akurasi Model MLP: {accuracy:.2f}%")
print("-" * 40 + "\n")

print(" Laporan Klasifikasi ")
print(classification_report(y_test, y_pred, target_names=le.classes_))
 

print("Membuat Visualisasi Grafik...")
cm = confusion_matrix(y_test, y_pred)
class_names = le.classes_ 

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names
)
plt.title('Matriks Kebingungan (Confusion Matrix)')
plt.ylabel('Nilai Aktual')
plt.xlabel('Nilai Prediksi')
plt.show()
