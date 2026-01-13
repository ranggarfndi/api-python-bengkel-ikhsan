import numpy as np
import joblib
import pandas as pd
import os
import json
import math
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ======================================================
# 1. MUAT MODEL & KONFIGURASI
# ======================================================
print("--- MEMULAI SERVER API ---")
try:
    model = joblib.load('model_spk.pkl')
    model_columns = joblib.load('kolom_model.pkl')
    print("✅ Model berhasil dimuat.")
except FileNotFoundError:
    print("❌ ERROR: File model tidak ditemukan.")
    exit()

# Bobot Kriteria (C1, C2 = Benefit | C3, C4 = Cost)
CRITERIA_WEIGHTS = {
    'c1_keluhan': {'weight': 0.40, 'type': 'benefit', 'label': 'C1. Jenis Keluhan'},
    'c2_harga':   {'weight': 0.30, 'type': 'benefit', 'label': 'C2. Harga'}, 
    'c3_jarak':   {'weight': 0.15, 'type': 'cost',    'label': 'C3. Jarak Tempuh'},
    'c4_usia':    {'weight': 0.15, 'type': 'cost',    'label': 'C4. Usia Kendaraan'}
}

# Data Paket Servis (Nama Baru)
ALTERNATIF_PERAWATAN = {
    'Ringan': [
        {'nama': 'Paket Cek Ringan',    'c1_keluhan': 1, 'c2_harga': 1, 'c3_jarak': 1, 'c4_usia': 1, 'harga_asli': 35000},
        {'nama': 'Paket Servis Hemat',  'c1_keluhan': 2, 'c2_harga': 1, 'c3_jarak': 1, 'c4_usia': 1, 'harga_asli': 65000},
        {'nama': 'Paket Harian',        'c1_keluhan': 2, 'c2_harga': 1, 'c3_jarak': 2, 'c4_usia': 2, 'harga_asli': 85000},
    ],
    'Sedang': [
        {'nama': 'Paket Standar',         'c1_keluhan': 4, 'c2_harga': 1, 'c3_jarak': 3, 'c4_usia': 3, 'harga_asli': 150000},
        {'nama': 'Paket Servis Bulanan',  'c1_keluhan': 3, 'c2_harga': 1, 'c3_jarak': 4, 'c4_usia': 4, 'harga_asli': 185000},
        {'nama': 'Paket Servis Lengkap',  'c1_keluhan': 4, 'c2_harga': 1, 'c3_jarak': 3, 'c4_usia': 3, 'harga_asli': 225000},
    ],
    'Tinggi': [
        {'nama': 'Paket Servis Besar',    'c1_keluhan': 5, 'c2_harga': 2, 'c3_jarak': 4, 'c4_usia': 4, 'harga_asli': 650000},
        {'nama': 'Paket Servis Total',    'c1_keluhan': 4, 'c2_harga': 3, 'c3_jarak': 4, 'c4_usia': 5, 'harga_asli': 950000},
        {'nama': 'Paket Overhaul',        'c1_keluhan': 5, 'c2_harga': 4, 'c3_jarak': 5, 'c4_usia': 5, 'harga_asli': 1800000},
    ]
}

# ======================================================
# 2. FUNGSI PERHITUNGAN COPRAS (DETAIL)
# ======================================================
def hitung_copras(alternatives_list, criteria_weights):
    criteria_keys = list(criteria_weights.keys())
    
    # Matriks X
    dm = np.array([[alt[key] for key in criteria_keys] for alt in alternatives_list])
    
    weights = np.array([cw['weight'] for cw in criteria_weights.values()])
    criteria_types = np.array([cw['type'] for cw in criteria_weights.values()])
    labels = [cw['label'] for cw in criteria_weights.values()]

    # Normalisasi R
    col_sums = dm.sum(axis=0)
    col_sums[col_sums == 0] = 1 
    norm_dm = dm / col_sums
    
    # Matriks D (Terbobot)
    weighted_dm = norm_dm * weights
    
    # S+ dan S-
    benefit_mask = (criteria_types == 'benefit')
    cost_mask = (criteria_types == 'cost')
    
    S_plus = weighted_dm[:, benefit_mask].sum(axis=1)
    S_minus = weighted_dm[:, cost_mask].sum(axis=1)
    
    # Hitung Qi (Bobot Relatif)
    sum_S_minus = S_minus.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_S_minus = 1 / S_minus
        inv_S_minus[np.isinf(inv_S_minus)] = 0
    sum_inv_S_minus = inv_S_minus.sum()
        
    term2 = 0
    if inv_S_minus.sum() > 0:
        term2 = sum_S_minus / (S_minus * inv_S_minus.sum())
    Q = S_plus + term2
    Q_max = Q.max()
    
    # Hitung Ui (Utilitas)
    if Q.max() > 0:
        U = (Q / Q.max()) * 100
    else:
        U = Q * 0

    results = []
    # Data Debug Lengkap untuk Laporan HTML
    full_debug = {
        'matrix_raw': dm.tolist(),
        'col_sums': col_sums.tolist(),
        'matrix_norm': norm_dm.tolist(),
        'matrix_weighted': weighted_dm.tolist(),
        'criteria_labels': labels,
        'weights': weights.tolist(),
        'alternatives_names': [alt['nama'] for alt in alternatives_list],
        'sum_S_minus': float(sum_S_minus),
        'sum_inv_S_minus': float(sum_inv_S_minus),
        'Q_max': float(Q_max)
    }

    for i, alt in enumerate(alternatives_list):
        results.append({
            'nama': alt['nama'],
            'harga_asli': alt['harga_asli'],
            'S_plus': float(S_plus[i]),
            'S_minus': float(S_minus[i]),
            'Q_i': float(Q[i]),
            'Utility (U_i)': float(U[i])
        })
        
    ranked_results = sorted(results, key=lambda x: x['Utility (U_i)'], reverse=True)
    return ranked_results, full_debug

# ======================================================
# 3. API ENDPOINT
# ======================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Ambil Input User
        nama_motor = data.get('nama_motor')
        usia_motor = data.get('usia_motor')
        jarak_tempuh = data.get('jarak_tempuh')
        gejala = data.get('gejala')
        
        if not all([nama_motor, usia_motor, jarak_tempuh, gejala]):
            return jsonify({'error': 'Data input tidak lengkap.'}), 400

        # --- PREPROCESSING KE FORMAT MODEL ---
        data_df = pd.DataFrame(columns=model_columns, index=[0])
        data_df = data_df.fillna(0)
        data_df['Usia Motor (tahun)'] = int(usia_motor)
        data_df['Jarak Tempuh (km)'] = int(jarak_tempuh)
        
        col_motor = f"Nama Motor_{nama_motor}"
        if col_motor in data_df.columns: data_df[col_motor] = 1
            
        col_gejala = f"Gejala Kerusakan_{gejala}"
        if col_gejala in data_df.columns: data_df[col_gejala] = 1

        # --- PREDIKSI AI ---
        hasil_prediksi_kelas = model.predict(data_df)[0]
        
        # Hitung Probabilitas (Untuk Rumus Entropy di Laporan)
        probs = model.predict_proba(data_df)[0] # Contoh: [0.1, 0.8, 0.1]
        
        # Hitung Entropy Dinamis dari Probabilitas Prediksi
        entropy_val = 0
        for p in probs:
            if p > 0: entropy_val += (-p * math.log2(p))
            
        # Siapkan data debug Decision Tree
        dt_debug = {
            'log_p1': math.log2(probs[0]) if probs[0] > 0 else 0, # Probabilitas Kelas 1
            'log_p2': math.log2(probs[1]) if probs[1] > 0 else 0, # Probabilitas Kelas 2
            'log_p3': math.log2(probs[2]) if probs[2] > 0 else 0, # Probabilitas Kelas 3 (jika ada)
            'entropy': entropy_val
        }

        # --- HITUNG COPRAS ---
        rekomendasi_paket = []
        copras_debug = {}
        
        if hasil_prediksi_kelas in ALTERNATIF_PERAWATAN:
            alternatives = ALTERNATIF_PERAWATAN[hasil_prediksi_kelas]
            full_ranking, copras_debug = hitung_copras(alternatives, CRITERIA_WEIGHTS)
            rekomendasi_paket = full_ranking[:3] # Ambil Top 3

        # --- GENERATE LAPORAN HTML ---
        try:
            # Data input rapi untuk ditampilkan di HTML
            input_display = {
                'motor': nama_motor,
                'usia': usia_motor,
                'km': jarak_tempuh,
                'gejala': gejala
            }
            
            with app.app_context():
                html_content = render_template(
                    'report.html', 
                    kategori=hasil_prediksi_kelas,
                    input_data=input_display,
                    rekomendasi=rekomendasi_paket,
                    debug=copras_debug,
                    dt_debug=dt_debug
                )
                with open("laporan_terbaru.html", "w", encoding="utf-8") as f:
                    f.write(html_content)
        except Exception as e:
            print(f"⚠️ Gagal generate laporan: {e}")

        # --- RETURN JSON ---
        return jsonify({
            'status': 'sukses',
            'hasil_diagnosa': {
                'kategori_perawatan': hasil_prediksi_kelas,
                'gejala_input': gejala,
                'motor_input': nama_motor
            },
            'rekomendasi_terbaik': rekomendasi_paket
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)