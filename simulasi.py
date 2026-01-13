import numpy as np
import math
import os
from jinja2 import Environment, FileSystemLoader

# ==============================================================================
# 1. KONFIGURASI BOBOT & DATA
# ==============================================================================
CRITERIA_WEIGHTS = {
    'c1_keluhan': {'weight': 0.40, 'type': 'benefit', 'label': 'C1. Jenis Keluhan'},
    'c2_harga':   {'weight': 0.30, 'type': 'benefit', 'label': 'C2. Harga'}, 
    'c3_jarak':   {'weight': 0.15, 'type': 'cost',    'label': 'C3. Jarak Tempuh'},
    'c4_usia':    {'weight': 0.15, 'type': 'cost',    'label': 'C4. Usia Kendaraan'}
}

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

# ==============================================================================
# 2. LOGIKA COPRAS
# ==============================================================================
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
    
    # Variabel Global
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
    
    # Ui
    if Q.max() > 0:
        U = (Q / Q.max()) * 100
    else:
        U = Q * 0

    results = []
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

# ==============================================================================
# 3. JALANKAN TES
# ==============================================================================
if __name__ == "__main__":
    print("=== TES LAPORAN LENGKAP (ANGKA ASLI) ===")
    
    # INPUT SIMULASI
    input_simulasi = "Sedang" 
    detail_input = {'motor': 'Honda Vario 150', 'usia': 5, 'km': 55000, 'gejala': 'Getaran CVT Parah'}
    
    print(f"Skenario: {input_simulasi}")
    
    # DATA DUMMY UNTUK HITUNGAN ENTROPY (DECISION TREE)
    # Misal: Dari 10 kasus serupa, distribusinya: 3 Ringan, 5 Sedang, 2 Tinggi
    total_kasus = 10
    prob_ringan = 3/10
    prob_sedang = 5/10
    prob_tinggi = 2/10
    
    # Hitung Entropy Manual untuk Ditampilkan
    entropy_val = 0
    for p in [prob_ringan, prob_sedang, prob_tinggi]:
        if p > 0: entropy_val += (-p * math.log2(p))
    
    dt_debug = {
        'total': total_kasus,
        'p1': prob_ringan, 'p2': prob_sedang, 'p3': prob_tinggi,
        'log_p1': math.log2(prob_ringan), 
        'log_p2': math.log2(prob_sedang), 
        'log_p3': math.log2(prob_tinggi),
        'entropy': entropy_val
    }

    if input_simulasi in ALTERNATIF_PERAWATAN:
        kandidat = ALTERNATIF_PERAWATAN[input_simulasi]
        hasil_ranking, data_debug = hitung_copras(kandidat, CRITERIA_WEIGHTS)

        try:
            file_loader = FileSystemLoader('templates')
            env = Environment(loader=file_loader)
            template = env.get_template('report.html')
            
            output_html = template.render(
                kategori=input_simulasi,
                input_data=detail_input,
                rekomendasi=hasil_ranking,
                debug=data_debug,
                dt_debug=dt_debug # <-- Kirim data entropy ke HTML
            )
            
            with open("laporan_lengkap_final.html", "w", encoding="utf-8") as f:
                f.write(output_html)
                
            print(f"✅ Laporan berhasil: laporan_lengkap_final.html")
        except Exception as e:
            print(f"❌ Error: {e}")