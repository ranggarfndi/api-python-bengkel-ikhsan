# 🔧 Sistem Informasi Manajemen Bengkel & Diagnosa AI

![Laravel]
![Python]

Aplikasi berbasis web yang mengintegrasikan sistem manajemen bengkel (CRUD) dengan kecerdasan buatan (Machine Learning). Sistem ini terdiri dari dua bagian utama: **Laravel** sebagai antarmuka pengguna & manajemen data, serta **Python** sebagai layanan API untuk prediksi kerusakan dan rekomendasi perawatan menggunakan metode **COPRAS**.


## 1. Prasyarat Sistem
Pastikan komputer Anda telah terinstal software berikut:
* **PHP** >= 8.1
* **Composer** (Manajer dependensi PHP)
* **Python** 3.x & **PIP**
* **MySQL** (XAMPP/Laragon/DBngin)
* **Node.js** (Opsional, untuk build aset jika mengubah CSS)

---

## 2. Panduan Instalasi & Menjalankan

Sistem ini membutuhkan **Dua Terminal** yang berjalan bersamaan (Satu untuk Laravel, Satu untuk Python).

### 🟢 Tahap A: Setup Python (API & AI)
Folder: `projek_api_spk`

1.  Buka terminal, arahkan ke folder API:
    ```bash
    cd projek_api_spk
    ```
2.  Install library yang dibutuhkan:
    ```bash
    pip install flask pandas scikit-learn numpy joblib
    ```
3.  **Jalankan Server Python:**
    ```bash
    python api.py
    ```
    ✅ *Output Sukses:* `Running on http://127.0.0.1:5000`

---

### 🔴 Tahap B: Setup Laravel (Aplikasi Utama)
Folder: `proyek_bengkel_spk`

1.  Buka terminal baru, arahkan ke folder Laravel:
    ```bash
    cd proyek_bengkel_spk
    ```
2.  Install dependensi project:
    ```bash
    composer install
    ```
3.  Salin konfigurasi environment:
    ```bash
    cp .env.example .env
    ```
4.  **Konfigurasi Database:**
    Buka file `.env`, sesuaikan nama database:
    ```env
    DB_DATABASE=db_bengkel_spk
    DB_USERNAME=root
    DB_PASSWORD=
    ```
5.  Generate Application Key:
    ```bash
    php artisan key:generate
    ```
6.  Migrasi Database & Isi Data Dummy Untuk Pelanggan dan Kendaraan Di Website (Seeder):
    ```bash
    php artisan migrate:fresh --seed
    ```
7.  **Jalankan Server Laravel:**
    ```bash
    php artisan serve
    ```
    ✅ *Output Sukses:* `Server running on http://127.0.0.1:8000`

---

## 3. Alur Kerja Website (User Flow)

Berikut adalah alur perjalanan pengguna dari awal hingga mendapatkan hasil diagnosa:

1.  **Login Admin:**
    * User mengakses `http://127.0.0.1:8000`.
    * Login menggunakan kredensial Admin (*admin@example.com*).
2.  **Dashboard Monitoring:**
    * Melihat statistik total servis, grafik distribusi kerusakan, dan tren gejala.
3.  **Manajemen Data:**
    * Admin mendaftarkan **Pelanggan** baru.
    * Admin menambahkan **Kendaraan** milik pelanggan tersebut.
4.  **Proses Predict (Inti Sistem):**
    * Admin masuk menu **Predict**.
    * Memilih motor pelanggan (Form otomatis mengisi usia motor).
    * Menginput gejala dan KM tempuh.
    * Klik **"Jalankan Predict"**.
    * *System Action:* Laravel mengirim data ke Python -> Python memproses diagnosa -> Python mengirim balik hasil JSON.
    * Hasil muncul dalam **Pop-up Modal** (Kategori Kerusakan + Rekomendasi Paket).
5.  **Pelaporan:**
    * Data otomatis tersimpan di menu **Riwayat**.
    * Admin dapat mengunduh laporan dalam bentuk **PDF/Excel**.

---

## 4. Fungsi & Tugas Setiap Halaman

### 🏠 Dashboard (`/dashboard`)
* **Fungsi:** Pusat informasi eksekutif.
* **Fitur:**
    * Kartu Statistik Berwarna (Total, Ringan, Sedang, Berat).
    * **Donut Chart (Chart.js):** Visualisasi persentase kategori kerusakan.
    * **Tabel Tren:** Top 5 gejala yang paling sering muncul.
    * List aktivitas servis terbaru.

### 🔍 Diagnosa (`/predict`)
* **Fungsi:** Halaman operasional utama untuk input data kerusakan.
* **Fitur Utama:**
    * **Auto-Calculate Age:** Script JS yang menghitung usia motor otomatis saat plat nomor dipilih.
    * **Integrasi API:** Mengirim request POST ke server Python.
    * **Modal Result:** Menampilkan hasil diagnosa tanpa reload halaman (menggunakan overlay).

### 📜 Riwayat (`/history`)
* **Fungsi:** Arsip digital seluruh aktivitas bengkel.
* **Fitur Utama:**
    * **DataTables:** Tabel canggih dengan fitur *Search* dan *Pagination*.
    * **Custom Sorting:** Dropdown untuk mengurutkan data (Terbaru/Terlama).
    * **Export Tools:** Tombol untuk download laporan ke **Excel**, **PDF**, atau **Print**.

### 👥 Pelanggan & Kendaraan (`/customers`, `/vehicles`)
* **Fungsi:** Manajemen data master (CRUD).
* **Fitur:**
    * Tambah/Edit/Hapus data pelanggan dan motor.
    * Visualisasi data dengan ikon dan badge status.
    * Pencarian data instan.

---

## 5. Bedah File Penting (Code Structure)

### 📂 Bagian Laravel (Backend & Frontend)

1.  **`app/Http/Controllers/PredictionController.php`**
    * **Peran:** Jembatan (Bridge) antara Laravel dan Python.
    * **Logika:** Menerima input form -> Validasi -> Menggunakan `Http::post` ke port 5000 -> Menerima respon JSON -> Menyimpan ke database -> Mengembalikan view.

2.  **`database/seeders/DatabaseSeeder.php`**
    * **Peran:** Generator Data Otomatis.
    * **Logika:** Membuat akun Admin, 15 Pelanggan palsu (Faker), dan Kendaraan secara acak untuk keperluan testing.

3.  **`resources/views/predict.blade.php`**
    * **Peran:** Antarmuka Diagnosa.
    * **Kode Kunci:**
        * JavaScript `change` event listener untuk menghitung umur motor.
        * Logika Blade `@if(isset($hasil))` untuk memunculkan Modal Pop-up.

4.  **`resources/views/history/index.blade.php`**
    * **Peran:** Tampilan Laporan.
    * **Kode Kunci:** Inisialisasi library **DataTables** dan ekstensi **Buttons** (Excel/PDF) dengan konfigurasi bahasa Indonesia.

### 🐍 Bagian Python (Diagnosa Logic)

1.  **`api.py`**
    * **Peran:** Server Flask.
    * **Logika:**
        * Endpoint `/predict` (POST).
        * Memuat model (`.pkl`).
        * Menerima JSON dari Laravel.
        * Melakukan prediksi klasifikasi.
        * Menjalankan perhitungan COPRAS.
        * Mengembalikan JSON ke Laravel.

2.  **`model_spk.pkl`**
    * **Peran:** Otak AI. File biner hasil training algoritma (C4.5/Naive Bayes/dll) yang menyimpan pola data lama.

---

## 6. Struktur Database

Sistem menggunakan database relasional (**MySQL**) dengan skema berikut:

| Tabel | Deskripsi | Relasi Utama |
| :--- | :--- | :--- |
| **`users`** | Menyimpan data login Admin. | - |
| **`customers`** | Data pemilik (Nama, HP, Alamat). | `hasMany` Vehicles |
| **`vehicles`** | Data motor (Plat, Merek, Tahun). | `belongsTo` Customer |
| **`prediction_histories`** | Data riwayat diagnosa (Input Gejala, Hasil Prediksi, Rekomendasi). | `belongsTo` Vehicle |

