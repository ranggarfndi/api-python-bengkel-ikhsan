import pandas as pd

nama_file = 'data_perawatan_motor_honda_100.csv'

try:
    # Coba baca file CSV, lewati baris yang error (seperti 'NB=...')
    df = pd.read_csv(nama_file, on_bad_lines='skip')
    
    # PERINTAH UTAMA:
    # Cetak daftar nama kolom yang dibaca oleh pandas
    print("--- Nama Kolom Aktual yang Dibaca Pandas ---")
    print(df.columns.to_list())
    print("---------------------------------------------")

except FileNotFoundError:
    print(f"Error: File {nama_file} tidak ditemukan.")
except Exception as e:
    print(f"Terjadi error: {e}")