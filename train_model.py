import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

NAMA_FILE_CSV = 'dataset-ikhsan-new.csv'
KOLOM_TARGET = 'Kelas Perawatan Kebutuhan'

def train_new_model():
    print(f"--- MEMULAI TRAINING MODEL BARU ---")
    print(f"1. Membaca data dari: {NAMA_FILE_CSV}...")
    
    try:
        # Baca CSV
        df = pd.read_csv(NAMA_FILE_CSV)
        print(f"   ✅ Data berhasil dibaca! Total baris: {len(df)}")
    except FileNotFoundError:
        print("   ❌ Error: File CSV tidak ditemukan.")
        return

    # 2. Pilih Fitur (Input) dan Target (Output)
    # Kita hanya mengambil kolom yang relevan untuk diagnosa
    fitur_cols = ['Nama Motor', 'Usia Motor (tahun)', 'Jarak Tempuh (km)', 'Gejala Kerusakan']
    
    print("2. Memilih fitur relevan...")
    try:
        X = df[fitur_cols]
        y = df[KOLOM_TARGET]
    except KeyError as e:
        print(f"   ❌ Error: Kolom {e} tidak ditemukan di CSV.")
        return

    # 3. Preprocessing (One-Hot Encoding)
    # Mengubah teks (Nama Motor, Gejala) menjadi angka biner otomatis
    print("3. Melakukan Encoding Data (Teks -> Angka)...")
    X_encoded = pd.get_dummies(X)
    
    # Simpan daftar nama kolom final agar API tidak bingung nanti
    final_columns = X_encoded.columns
    print(f"   Total fitur setelah encoding: {len(final_columns)}")

    # 4. Latih Model Decision Tree
    print("4. Melatih Model AI (Decision Tree)...")
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_encoded, y)
    
    # 5. Simpan Model
    print("5. Menyimpan file model (.pkl)...")
    joblib.dump(model, 'model_spk.pkl')
    joblib.dump(final_columns, 'kolom_model.pkl')
    
    print("\n✅ SUKSES! Model 'model_spk.pkl' telah diperbarui dengan data CSV baru.")

if __name__ == "__main__":
    train_new_model()