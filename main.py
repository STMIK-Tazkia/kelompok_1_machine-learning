import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

try:
    # --- 1. Muat Dataset ---
    file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'
    data = pd.read_csv(file_path)

    print(f"Dataset berhasil dimuat. Jumlah baris: {len(data)}, Jumlah kolom: {len(data.columns)}")
    print("-" * 30)

    # --- 2. Pisahkan Fitur (X) dan Target (y) ---
    # 'NObeyesdad' adalah kolom target yang ingin kita prediksi
    X = data.drop('NObeyesdad', axis=1)
    y = data['NObeyesdad']

    # --- 3. Encode Variabel Target (y) ---
    # Model machine learning memerlukan target dalam bentuk angka, bukan teks
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Simpan nama kelas untuk laporan akhir
    class_names = le.classes_

    # --- 4. Identifikasi Jenis Fitur ---
    # Ini penting untuk memberitahu ColumnTransformer cara memproses setiap kolom
    numerical_features = [
        'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'
    ]
    
    categorical_features = [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
        'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ]

    # --- 5. Buat Pipeline Pra-pemrosesan ---
    
    # Transformer untuk fitur numerik:
    # StandardScaler akan mengubah skala data (misal: Age, Weight) agar memiliki mean 0 dan std dev 1
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Transformer untuk fitur kategorikal:
    # OneHotEncoder akan mengubah kolom seperti 'Gender' (Male/Female) menjadi
    # dua kolom biner (0 atau 1)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gabungkan kedua transformer menggunakan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Jaga kolom lain jika ada (meski dalam kasus ini tidak ada)
    )

    # --- 6. Buat Pipeline Model Lengkap ---
    # Pipeline ini akan secara otomatis:
    # 1. Menerapkan 'preprocessor' ke data X
    # 2. Melatih 'classifier' (GaussianNB) pada data yang sudah diproses
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])

    # --- 7. Bagi Data Latih dan Data Uji ---
    # Kita gunakan 80% data untuk melatih model dan 20% untuk mengujinya
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # --- 8. Latih Model ---
    print("Memulai pelatihan model Naive Bayes...")
    model.fit(X_train, y_train)
    print("Pelatihan model selesai.")
    print("-" * 30)

    # --- 9. Evaluasi Model ---
    # Lakukan prediksi pada data uji
    y_pred = model.predict(X_test)

    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Akurasi Model: {accuracy * 100:.2f}%')
    print("\nLaporan Klasifikasi (Classification Report):")
    
    # Tampilkan laporan lengkap (precision, recall, f1-score)
    # Gunakan 'target_names' untuk menampilkan label asli, bukan angka
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # --- 10. Contoh Prediksi Data Baru ---
    print("-" * 30)
    print("Contoh Prediksi Data Baru:")
    
    # Ambil satu baris data dari dataset asli untuk contoh
    # (Kita gunakan .iloc[10:11] untuk mengambil baris ke-10)
    sample_data = X.iloc[10:11].copy()
    
    print("\nData yang akan diprediksi:")
    print(sample_data)
    
    # Lakukan prediksi
    sample_prediction_encoded = model.predict(sample_data)
    
    # Ubah kembali hasil prediksi (angka) ke label asli (teks)
    sample_prediction_label = le.inverse_transform(sample_prediction_encoded)
    
    print(f"\nHasil Prediksi: {sample_prediction_label[0]}")
    print(f"Label Sebenarnya: {data.iloc[10]['NObeyesdad']}")

except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan.")
except Exception as e:
    print(f"Terjadi error: {e}")