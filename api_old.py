import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# 1. Impor fungsi preprocessing kita
from prepare_data import standarisasi_gejala

# 2. Inisialisasi Aplikasi Flask
app = Flask(__name__)

# 3. Muat Model & Kolom (HANYA SEKALI saat server dimulai)
print("Memuat model dan daftar kolom...")
try:
    model = joblib.load('model_spk.pkl')
    model_columns = joblib.load('kolom_model.pkl')
    print("Model dan kolom berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'model_spk.pkl' atau 'kolom_model.pkl' tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'train_model.py' terlebih dahulu.")
    exit()


# ======================================================
# TAHAP 3.A: DATA MANUAL UNTUK MATRIKS COPRAS

# 1. TENTUKAN BOBOT KEPENTINGAN (Ini adalah inti penelitian Anda!)
# Bobot ini harus Anda justifikasi di skripsi Anda.
# Total bobot harus 1.0 (100%)
# 'cost' = makin kecil makin bagus (minimize)
# 'benefit' = makin besar makin bagus (maximize)
CRITERIA_WEIGHTS = {
    'harga':        {'weight': 0.4, 'type': 'cost'},
    'waktu':        {'weight': 0.3, 'type': 'cost'},
    'kompleksitas': {'weight': 0.3, 'type': 'benefit'}
}

# 2. DEFINISIKAN ALTERNATIF UNTUK SETIAP KELAS PERAWATAN
# Ini adalah 'database' perbandingan Anda.
# Nilai (1-5): 1=Sangat Buruk, 5=Sangat Baik (atau sebaliknya untuk 'cost')
# Harga: 1=Sangat Mahal, 5=Sangat Murah
# Waktu: 1=Sangat Lama, 5=Sangat Cepat
# Kompleksitas: 1=Sangat Sedikit, 5=Sangat Lengkap/Menyeluruh
ALTERNATIF_PERAWATAN = {
    'Ringan': [
        {'nama': 'Paket Ganti Oli', 'harga': 5, 'waktu': 5, 'kompleksitas': 2, 'harga_rupiah': 65000},
        {'nama': 'Paket Servis Rutin', 'harga': 4, 'waktu': 4, 'kompleksitas': 3, 'harga_rupiah': 85000},
    ],
    'Sedang': [
        {'nama': 'Paket Servis CVT', 'harga': 4, 'waktu': 4, 'kompleksitas': 4, 'harga_rupiah': 150000},
        {'nama': 'Paket Servis Injeksi & Kelistrikan', 'harga': 3, 'waktu': 3, 'kompleksitas': 5, 'harga_rupiah': 225000},
        {'nama': 'Paket Servis Berkala (Rem & Rantai)', 'harga': 5, 'waktu': 5, 'kompleksitas': 3, 'harga_rupiah': 120000},
    ],
    'Tinggi': [
        {'nama': 'Paket Servis Besar (Turun Mesin Sebagian)', 'harga': 3, 'waktu': 2, 'kompleksitas': 4, 'harga_rupiah': 650000},
        {'nama': 'Paket Overhaul (Turun Mesin Total)', 'harga': 1, 'waktu': 1, 'kompleksitas': 5, 'harga_rupiah': 1800000},
    ]
}

def hitung_copras(alternatives_list, criteria_weights):
    # 0. Ekstrak data
    criteria_keys = list(criteria_weights.keys())
    dm = np.array([
        [alt[key] for key in criteria_keys] 
        for alt in alternatives_list
    ])
    
    weights = np.array([cw['weight'] for cw in criteria_weights.values()])
    criteria_types = np.array([cw['type'] for cw in criteria_weights.values()])

    # 1. Normalisasi
    col_sums = dm.sum(axis=0)
    norm_dm = dm / col_sums
    
    # 2. Pembobotan
    weighted_dm = norm_dm * weights
    
    # 3. Hitung S+ dan S-
    benefit_mask = (criteria_types == 'benefit')
    cost_mask = (criteria_types == 'cost')
    S_plus = weighted_dm[:, benefit_mask].sum(axis=1)
    S_minus = weighted_dm[:, cost_mask].sum(axis=1)
    
    # 4. Hitung Qi
    if np.any(S_minus == 0):
        Q = S_plus - S_minus
    else:
        Q = S_plus + (S_minus.sum() * (1 / S_minus).sum())**-1 / S_minus

    # 5. Hitung Ui
    Q_max = Q.max()
    U = (Q / Q_max) * 100
    
    # 6. Susun Hasil (SEKARANG TERMASUK HARGA RUPIAH)
    results = []
    for i, alt in enumerate(alternatives_list):
        results.append({
            'nama': alt['nama'],
            'harga_rupiah': alt['harga_rupiah'], # <-- PENTING: Kita sertakan harga asli
            'S_plus': S_plus[i],
            'S_minus': S_minus[i],
            'Q_i': Q[i],
            'Utility (U_i)': U[i]
        })
        
    ranked_results = sorted(results, key=lambda x: x['Utility (U_i)'], reverse=True)
    return ranked_results
# ======================================================


# 4. Definisikan API Endpoint '/predict'
# 'methods=['POST']' berarti endpoint ini hanya menerima data (TIDAK bisa diakses dari browser)
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint utama untuk menerima data, menjalankan Decision Tree,
    DAN menjalankan COPRAS.
    """
    try:
        # 1. Ambil data JSON yang dikirim oleh Laravel
        data = request.get_json()

        # 2. Ambil nilai input dari JSON
        nama_motor = data.get('nama_motor')
        usia_motor = data.get('usia_motor')
        jarak_tempuh = data.get('jarak_tempuh')
        gejala = data.get('gejala')
        
        if not all([nama_motor, usia_motor, jarak_tempuh, gejala]):
            return jsonify({'error': 'Data tidak lengkap.'}), 400

        # 3. Proses Input agar Sesuai Format Model
        data_df = pd.DataFrame(columns=model_columns, index=[0])
        data_df = data_df.fillna(0)
        data_df['Usia Motor (tahun)'] = int(usia_motor)
        data_df['Jarak Tempuh (km)'] = int(jarak_tempuh)
        
        gejala_kategori = standarisasi_gejala(gejala)
        col_gejala = f"Gejala_Terkategori_{gejala_kategori}"
        if col_gejala in data_df.columns:
            data_df[col_gejala] = 1

        col_motor = f"Nama Motor_{nama_motor}"
        if col_motor in data_df.columns:
            data_df[col_motor] = 1
        
        # 4. Lakukan Prediksi Decision Tree (Filter)
        prediksi = model.predict(data_df)
        hasil_prediksi_kelas = prediksi[0] # Hasil: "Ringan", "Sedang", atau "Tinggi"

        # ================================================
        # == TAHAP 3.C: INTEGRASI COPRAS DIMULAI DI SINI ==
        # ================================================
        
        peringkat_copras = [] # Buat list kosong
        
        # Periksa apakah kelas prediksi ada di data alternatif kita
        if hasil_prediksi_kelas in ALTERNATIF_PERAWATAN:
            # Ambil alternatif untuk kelas tersebut (misal, semua paket 'Sedang')
            alternatives = ALTERNATIF_PERAWATAN[hasil_prediksi_kelas]
            
            # Panggil fungsi COPRAS yang sudah kita buat!
            peringkat_copras = hitung_copras(alternatives, CRITERIA_WEIGHTS)
        
        # ================================================
        # == INTEGRASI COPRAS SELESAI ==
        # ================================================

        # 5. Kembalikan Hasil sebagai JSON (SUDAH TERMASUK COPRAS)
        return jsonify({
            'status': 'sukses',
            'input_data': data,
            'input_terstandarisasi': {
                'gejala_kategori': gejala_kategori,
            },
            'prediksi_kelas_perawatan': hasil_prediksi_kelas,
            'rekomendasi_paket_copras': peringkat_copras # <-- HASIL BARU!
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 5. Jalankan Server API
if __name__ == '__main__':
    # 'port=5000' adalah port default untuk Flask.
    # Laravel akan "bicara" dengan http://localhost:5000
    app.run(debug=True, port=5000)


# ======================================================
# BLOK TES UNTUK TAHAP 3.B
# ======================================================
# if __name__ == '__main__':
#     # Uji fungsi COPRAS secara terpisah
#     print("\n--- Menguji Fungsi COPRAS (Tahap 3.B) ---")
    
#     # Ambil data manual kita untuk kelas 'Sedang'
#     test_alternatives = ALTERNATIF_PERAWATAN['Sedang']
#     test_weights = CRITERIA_WEIGHTS
    
#     print("Menghitung peringkat untuk kelas 'Sedang':")
#     peringkat_sedang = hitung_copras(test_alternatives, test_weights)
    
#     # Cetak hasilnya dengan rapi
#     import json
#     print(json.dumps(peringkat_sedang, indent=2))
    
#     print("\n--- Menjalankan Server Flask API ---")
#     # Jalankan server Flask
#     app.run(debug=True, port=5000)