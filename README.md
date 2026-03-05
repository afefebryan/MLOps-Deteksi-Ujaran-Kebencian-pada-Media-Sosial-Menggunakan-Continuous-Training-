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
