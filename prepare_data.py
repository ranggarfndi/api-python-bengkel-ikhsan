import pandas as pd
import re # Akan kita gunakan untuk membersihkan data

def standarisasi_gejala(teks):
    """
    Fungsi ini mengambil teks gejala mentah dan mengubahnya 
    menjadi kategori yang bersih dan standar.
    """
    teks = str(teks).lower() # Ubah ke huruf kecil

    # Kelompok 1: Masalah Mesin (Performa)
    if any(kata in teks for kata in ['tarikan berat', 'tarikan lambat']):
        return 'performa_tarikan_berat'
    if 'knalpot berasap' in teks or 'asap knalpot' in teks:
        return 'mesin_berasap'
    if 'mesin panas' in teks or 'overheat' in teks:
        return 'mesin_overheat'
    
    # Kelompok 2: Masalah Mesin (Suara)
    if any(kata in teks for kata in ['suara kasar', 'bunyi aneh', 'klotok', 'berisik', 'suara mesin kasar']):
        return 'suara_mesin_kasar'

    # Kelompok 3: Masalah Kelistrikan
    if any(kata in teks for kata in ['aki soak', 'aki tekor', 'starter sulit', 'starter mati', 'tidak bisa di-starter']):
        return 'kelistrikan_aki_starter'
    if 'lampu redup' in teks or 'lampu mati' in teks:
        return 'kelistrikan_lampu'
    # Menangkap semua jenis "MIL" (Malfunction Indicator Lamp)
    if 'mil' in teks: 
        return 'kelistrikan_mil_menyala'

    # Kelompok 4: Masalah Transmisi & Rangka
    if 'oli bocor' in teks:
        return 'oli_bocor'
    if any(kata in teks for kata in ['cvt', 'gredeg', 'v-belt', 'roller']):
        return 'cvt_bermasalah'
    if 'kopling' in teks:
        return 'kopling_bermasalah'
    if 'rantai' in teks:
        return 'rantai_bermasalah'
    if 'rem kurang pakem' in teks or 'rem blong' in teks:
        return 'rem_bermasalah'
    if 'suspensi' in teks or 'shock' in teks:
        return 'suspensi_bermasalah'
    if 'getaran' in teks: # Menangkap 'mesin bergetar' dan 'getaran setang'
        return 'getaran_rangka'
        
    # Kelompok 5: Normal (Tidak ada gejala)
    if 'normal' in teks:
        return 'normal'
        
    # Jika tidak ada di atas, kategorikan sebagai 'lain'
    return 'gejala_lain'


def jalankan_preprocessing(file_csv):
    """
    Fungsi utama untuk memuat, membersihkan, dan 
    mentransformasi data.
    """
    # 1. Muat Data
    try:
        # !!! PERUBAHAN DI SINI !!!
        # Kita tambahkan 'sep=';' untuk memberitahu pandas 
        # agar menggunakan titik koma sebagai pemisah.
        # Saya juga menambahkan encoding='utf-8' untuk jaga-jaga.
        df = pd.read_csv(file_csv, on_bad_lines='skip', sep=';', encoding='utf-8')
        
    except FileNotFoundError:
        print(f"Error: File {file_csv} tidak ditemukan.")
        return None, None
    except Exception as e:
        print(f"Error saat membaca file CSV: {e}")
        print("Pastikan file tidak sedang terbuka di Excel.")
        return None, None

    # 2. Pembersihan Awal
    # Hapus baris duplikat jika ada
    df = df.drop_duplicates()
    
    # 3. Pisahkan Fitur (X) dan Target (y) SEBELUM encoding
    # Target kita adalah apa yang ingin kita prediksi
    try:
        y = df['Kelas Perawatan Kebutuhan'] # 
        
        # Fitur adalah data input yang kita gunakan untuk memprediksi
        fitur_awal = ['Nama Motor', 'Usia Motor (tahun)', 'Jarak Tempuh (km)', 'Gejala Kerusakan'] # 
        X_mentah = df[fitur_awal].copy()
        
    except KeyError as e:
        print(f"Error: Kolom {e} tidak ditemukan di file CSV.")
        print("Pastikan nama kolom di file CSV Anda sudah benar.")
        print("Nama kolom yang dibaca pandas:", df.columns.to_list())
        return None, None

    # 4. Feature Engineering (Standardisasi Gejala)
    X_mentah['Gejala_Terkategori'] = X_mentah['Gejala Kerusakan'].apply(standarisasi_gejala)
    X_mentah = X_mentah.drop(columns=['Gejala Kerusakan'])
    
    # 5. Encoding (Mengubah Teks menjadi Angka)
    kolom_kategorikal = ['Nama Motor', 'Gejala_Terkategori']
    X_encoded = pd.get_dummies(X_mentah, columns=kolom_kategorikal, drop_first=True)
    
    print("--- Proses Preprocessing Selesai (Versi Perbaikan 'sep=;') ---")
    print("\nContoh 5 baris data X (Fitur) yang sudah di-encode:")
    print(X_encoded.head())
    print(f"\nTotal Fitur (kolom) setelah encoding: {len(X_encoded.columns)}")
    
    print("\nContoh 5 baris data y (Target):")
    print(y.head())
    
    return X_encoded, y

# --- JALANKAN PROSES ---
if __name__ == "__main__":
    # Pastikan file 'data_perawatan_motor_honda_100.csv' 
    # ada di folder yang sama dengan script ini.
    nama_file = 'data_perawatan_motor_honda_100.csv'
    X_siap_latih, y_siap_latih = jalankan_preprocessing(nama_file)

    if X_siap_latih is not None:
        # Di tahap selanjutnya, kita akan gunakan X_siap_latih dan y_siap_latih
        # untuk melatih model Decision Tree.
        print("\nData Anda sekarang bersih dan siap untuk Tahap 1.C (Pelatihan Model).")