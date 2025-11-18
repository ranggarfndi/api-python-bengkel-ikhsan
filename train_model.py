import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # Library untuk menyimpan model ML Anda

# 1. Impor fungsi preprocessing dari file Anda sebelumnya
from prepare_data import jalankan_preprocessing

print("--- TAHAP 1.C: Pelatihan Model Dimulai ---")

# 2. Muat dan siapkan data
nama_file = 'data_perawatan_motor_honda_100.csv'
X, y = jalankan_preprocessing(nama_file)

if X is None or y is None:
    print("Gagal memuat data. Proses pelatihan dibatalkan.")
    exit()

# 3. Bagi Data: 80% untuk Latihan, 20% untuk Pengujian
# 'test_size=0.2' berarti 20% data akan disisihkan untuk pengujian.
# 'random_state=42' memastikan hasil pembagian data selalu sama 
# setiap kali skrip dijalankan (penting untuk reproduktifitas).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nData dibagi: {len(X_train)} baris untuk latihan, {len(X_test)} baris untuk pengujian.")

# 4. Inisialisasi dan Latih Model
# Kita gunakan 'criterion="entropy"' agar mirip dengan algoritma C4.5
# yang menggunakan Information Gain, seperti disebut di proposal [cite: 196-197].
model = DecisionTreeClassifier(criterion="entropy", random_state=42)

print("Melatih model Decision Tree...")
# 
model.fit(X_train, y_train)
print("Model berhasil dilatih.")

# 5. Evaluasi Model
# Mari kita lihat seberapa baik performa model kita pada data 
# yang BELUM PERNAH ia lihat sebelumnya (data pengujian).
print("\nMengevaluasi performa model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy * 100:.2f}%")

# Laporan ini menunjukkan presisi dan recall, 
# seperti yang disebutkan di proposal Anda [cite: 91]
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# 6. Simpan Model dan Kolom
# Ini adalah langkah paling penting untuk API kita.
# Kita simpan model yang sudah dilatih ke sebuah file.
nama_file_model = 'model_spk.pkl'
joblib.dump(model, nama_file_model)
print(f"\nModel telah disimpan ke file: {nama_file_model}")

# Kita juga HARUS menyimpan daftar kolom yang digunakan untuk melatih model.
# API kita nanti perlu tahu urutan kolom ini.
nama_file_kolom = 'kolom_model.pkl'
model_columns = X.columns.to_list()
joblib.dump(model_columns, nama_file_kolom)
print(f"Daftar kolom model telah disimpan ke file: {nama_file_kolom}")

print("\n--- TAHAP 1.C Selesai ---")