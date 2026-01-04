import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 1. Load Data
df = pd.read_csv('Raisin.csv')
X = df.drop('Class', axis=1).values # Mengambil 7 fitur numerik

# 2. Preprocessing (Penting: Scaling agar nilai jomplang tidak merusak gradien)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data untuk training dan validasi
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# 3. Arsitektur Autoencoder
input_dim = X_train.shape[1] # 7 fitur

# Encoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(5, activation='relu')(input_layer)
bottleneck = Dense(3, activation='relu')(encoder) # Kompresi menjadi 3 fitur

# Decoder
decoder = Dense(5, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='linear')(decoder) # Output 7 fitur

# Gabungkan Model
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# 4. Training
history = autoencoder.fit(
    X_train, X_train, # Targetnya adalah dirinya sendiri (Self-supervised)
    epochs=100,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=0
)

# 5. Visualisasi Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (Reconstruction Error)')
plt.legend()
plt.show()

print("Autoencoder selesai dilatih!")

#

