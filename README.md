# Sistem Deteksi Ujaran Kebencian pada Media Sosial dengan Continuous Training pada Data Dinamis

## 1. Tujuan Proyek

Proyek ini bertujuan untuk membangun pipeline deteksi ujaran kebencian pada media sosial dengan pendekatan **Continuous Training**. Proyek ini mencakup:

1. Eksplorasi dan preprocessing dataset awal yang sudah berlabel (Kaggle) untuk pelatihan model dasar.  
2. Akuisisi data baru secara berkala dari Reddit menggunakan mekanisme batch-based ingestion yang mensimulasikan karakteristik data streaming.  
3. Penerapan mekanisme retraining otomatis berbasis trigger, seperti penurunan F1-score atau perubahan distribusi data (KL Divergence).  
4. Pengelolaan versi model dan metrik performa menggunakan MLflow, untuk memastikan model tetap adaptif terhadap perubahan pola bahasa di media sosial.  
5. Penyediaan endpoint inferensi sederhana (FastAPI) untuk memprediksi komentar baru secara terukur.  

Dengan demikian, proyek ini fokus pada **adaptasi model terhadap data dinamis** sekaligus menyediakan pipeline yang jelas dan dapat direplikasi.

---
## 2. Struktur Direktori Proyek

```text
project/
│
├── data/             # Dataset awal dan data hasil scraping
├── models/           # Model yang sudah dilatih dan versi model
├── notebooks/        # Notebook eksplorasi data dan percobaan model
├── src/              # Script Python utama (preprocessing, training, inference)
├── config/           # File konfigurasi, parameter, dan environment
│
├── requirements.txt  # Daftar library yang diperlukan
├── README.md         # Dokumentasi proyek
└── .gitignore        # File dan folder yang diabaikan Git
```
---
## 3. Cara Menjalankan Codespaces

1. Pastikan sudah membuat Codespace untuk repository ini.  
2. Buka Codespace melalui GitHub:  
   a. Klik tombol **Code → Codespaces → Open in browser**  
   b. Atau **Open in VS Code** jika ingin pakai VS Code lokal.  
3. Tunggu hingga container selesai build (rebuild pertama bisa beberapa menit).  
4. Pastikan kernel Python yang dipakai sesuai `.devcontainer` (misal **Python 3.11**).  
5. Buka terminal di Codespace dan install dependencies (jika belum otomatis) dengan perintah:

```bash
pip install -r requirements.txt
```
## 4. Menjalankan Pipeline Pengumpulan Data

Pipeline ini digunakan untuk mengambil data dari Reddit, melakukan preprocessing, dan memberikan label otomatis pada data yang dikumpulkan. Seluruh proses dijalankan secara terintegrasi melalui satu file utama, yaitu `run_pipeline.py`.

### 4.1 Langkah-langkah

1. Masuk ke direktori source code (jika menggunakan struktur `src/`):

   ```bash
   cd src
   ```

2. Jalankan pipeline:

   ```bash
   python run_pipeline.py
   ```

### 4.2 Alur Pipeline

Pipeline akan menjalankan beberapa tahap secara berurutan:

| Tahap | Deskripsi |
|-------|-----------|
| **Extract** | Mengambil data komentar dari Reddit berdasarkan subreddit yang telah ditentukan. |
| **Transform** | Membersihkan teks, menghapus duplikasi, filtering user, serta membuat embedding. |
| **Load** | Memuat dataset hasil preprocessing. |
| **Labeling** | Memberikan label otomatis (**Neutral**, **Offensive**, **Hateful**) menggunakan model zero-shot. |

### 4.3 Output yang Dihasilkan

Setelah pipeline dijalankan, akan dihasilkan dua jenis file:

- **Raw Data** → disimpan di folder `data/raw/`
- **Processed Data** → disimpan di folder `data/processed/`

Setiap file menggunakan format penamaan:

```
<nama_file>_V<versi>_<tanggal>.csv
```

**Contoh:**

```
reddit_raw_comments_V1_2026-04-04.csv
reddit_clean_comments_V1_2026-04-04.csv
```

Sistem ini memastikan bahwa data tidak akan tertimpa saat pipeline dijalankan berulang kali.

## 5. Data Versioning dengan DVC

### 5.1 Inisialisasi DVC

DVC diinisialisasi pada repository untuk memungkinkan pelacakan data secara terpisah dari Git:

```
dvc init
```

---

### 5.2 Tracking Dataset Awal

Dataset awal dilacak menggunakan DVC agar tidak disimpan langsung di Git:

```
dvc add data/raw/dataset.csv
```

File `.dvc` yang dihasilkan akan di-commit ke Git, sementara file dataset asli akan diabaikan oleh Git melalui `.gitignore`.

---

### 5.3 Simulasi Continual Learning

Dataset diperbarui dengan menjalankan proses ingestion untuk mensimulasikan penambahan data baru:

```
python ingest_data.py
```

Proses ini menghasilkan data tambahan yang merepresentasikan skenario continual learning.

---

### 5.4 Versioning Dataset

Setelah dataset diperbarui, dilakukan pelacakan ulang menggunakan DVC:

```
dvc add data/raw/dataset.csv
```

Perubahan ini akan menghasilkan hash baru pada file `.dvc`, yang menandakan adanya versi dataset yang berbeda.

---

### 5.5 Audit dan Perbandingan Versi Data

Perbedaan antara versi dataset lama dan baru dapat dianalisis menggunakan:

```
dvc diff
```

Perintah ini memungkinkan identifikasi perubahan metadata dan memastikan bahwa setiap versi dataset tercatat dengan baik.

---

### 5.7 Tujuan Pengunaan DVC

DVC memungkinkan pelacakan versi dataset secara efisien tanpa membebani repository Git. Dengan pendekatan ini, setiap perubahan data dapat ditelusuri, direproduksi, dan dibandingkan antar versi, sehingga mendukung praktik pengembangan machine learning yang lebih terstruktur dan reproducible.
