# Sistem Deteksi Ujaran Kebencian pada Media Sosial dengan Continuous Training pada Data Dinamis

## 1. Overview Proyek

### 1.1 Tujuan Proyek
Proyek ini bertujuan untuk membangun pipeline deteksi ujaran kebencian pada media sosial dengan pendekatan **Continuous Training**. Proyek ini mencakup:

1. Eksplorasi dan preprocessing dataset awal yang sudah berlabel (Kaggle) untuk pelatihan model dasar.  
2. Akuisisi data baru secara berkala dari Reddit menggunakan mekanisme batch-based ingestion yang mensimulasikan karakteristik data streaming.  
3. Penerapan mekanisme retraining otomatis berbasis trigger, seperti penurunan F1-score 
4. Pengelolaan versi model dan metrik performa menggunakan MLflow, untuk memastikan model tetap adaptif terhadap perubahan pola bahasa di media sosial.  
5. Penyediaan endpoint inferensi sederhana (FastAPI) untuk memprediksi komentar baru secara terukur.  

Dengan demikian, proyek ini fokus pada **adaptasi model terhadap data dinamis** sekaligus menyediakan pipeline yang jelas dan dapat direplikasi.


### 1.2 Dataset yang Digunakan

Model pada proyek ini dilatih menggunakan dataset **tdavidson_hate_speech_v0_clean_train**, yang merupakan hasil pembersihan dari dataset hate speech milik Davidson et al. Dataset ini berisi kumpulan teks media sosial berbahasa Inggris yang telah diberi label untuk tugas klasifikasi ujaran kebencian.

Dataset digunakan sebagai data awal (baseline dataset) untuk membangun model pertama sebelum sistem menerima data baru dari proses ingestion Reddit. Selama proses continuous training, data baru yang diperoleh akan digabungkan dengan dataset yang sudah ada untuk melakukan retraining dan memperbarui model secara berkala.

Kolom utama yang digunakan dalam proses pelatihan adalah:

| Kolom | Deskripsi                                        |
| ----- | ------------------------------------------------ |
| text  | Teks komentar atau unggahan yang akan dianalisis |
| label | Label klasifikasi (harmful atau neutral)         |

Dataset ini disimpan dan dikelola menggunakan DVC sehingga setiap perubahan data dapat dilacak dan direproduksi selama proses pengembangan maupun retraining model.

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

### 5.6 Tujuan Pengunaan DVC

DVC memungkinkan pelacakan versi dataset secara efisien tanpa membebani repository Git. Dengan pendekatan ini, setiap perubahan data dapat ditelusuri, direproduksi, dan dibandingkan antar versi, sehingga mendukung praktik pengembangan machine learning yang lebih terstruktur dan reproducible.

## 6. Model Aktif untuk Inferensi

Model yang saat ini digunakan untuk proses inferensi adalah model terbaik hasil eksperimen yang dikelola menggunakan MLflow Model Registry.

---

### 6.1 Model Terpilih

| Atribut  | Detail                             |
|----------|------------------------------------|
| Model    | LinearSVC                          |
| Run      | Phase2_LinearSVC_run02             |
| Dataset  | tdavidson_hate_speech_v0_clean (v0) |
| F1 Macro | 0.9200                             |
| Accuracy | 0.9521                             |

Model ini dipilih sebagai model aktif karena memiliki performa terbaik dibandingkan seluruh eksperimen yang dilakukan, khususnya pada metrik **F1 Macro** yang menjadi fokus utama dalam menangani ketidakseimbangan kelas pada kasus klasifikasi ujaran kebencian.

---

### 6.2 Alasan Pemilihan Model

Pemilihan model didasarkan pada hasil evaluasi berikut:

```
              run_name  accuracy  f1_macro  f1_weighted
Phase2_LinearSVC_run02  0.952147  0.920005     0.953421
      Phase1_LinearSVM  0.952007  0.919937     0.953329
        Phase1_SGD_SVM  0.950463  0.919511     0.952403
Phase2_LinearSVC_run03  0.950182  0.915354     0.951135
             Phase1_LR  0.944850  0.910943     0.947156
Phase2_LinearSVC_run04  0.948077  0.910371     0.948681
             Phase1_RF  0.949481  0.908655     0.948898
Phase2_LinearSVC_run01  0.943026  0.908431     0.945521
         Phase1_SGD_LR  0.940219  0.904554     0.943003
Phase2_LinearSVC_run05  0.944289  0.902129     0.944460
```

`Phase2_LinearSVC_run02` menempati posisi tertinggi pada metrik **F1 Macro**, yang menunjukkan performa paling baik dalam menangani distribusi kelas yang tidak seimbang. Nilai **accuracy** yang tinggi juga mengonfirmasi bahwa model tetap stabil secara keseluruhan.

---

### 6.3 Status Model

Model telah didaftarkan ke dalam MLflow Model Registry dengan stage berikut:

```
Staging
```

Stage ini menunjukkan bahwa model telah siap digunakan dalam proses inferensi dan dapat dipromosikan ke tahap **Production** setelah melalui validasi lebih lanjut.

## 7. Menjalankan Sistem dengan Docker Compose

Sistem ini terdiri dari dua layanan yang dijalankan secara bersamaan menggunakan Docker Compose:

| Layanan | Deskripsi | Port |
|---|---|---|
| `mlflow-server` | MLflow Tracking Server + Model Registry (SQLite) | `5000` |
| `api-service` | REST API inferensi model (FastAPI) | `8000` |

---

### 7.1 Prasyarat

Pastikan sudah terinstall:
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (sudah termasuk di Docker Desktop)

Verifikasi instalasi:
```bash
docker --version
docker compose version
```

---

### 7.2 Jalankan Seluruh Sistem

Jalankan semua layanan dengan satu perintah dari direktori root proyek:

```bash
docker compose up -d
```

Docker akan secara otomatis:
1. Build image `api-service` dari `Dockerfile`
2. Pull image `mlflow-server`
3. Membuat network `mlops-network`
4. Membuat volume persisten `mlflow-db` dan `mlflow-artifacts`
5. Menjalankan `mlflow-server` terlebih dahulu, lalu `api-service` setelah server siap

---

### 7.3 Verifikasi Status Layanan

Cek status seluruh container yang berjalan:

```bash
docker compose ps
```

Output yang diharapkan:
```
NAME             IMAGE           STATUS                   PORTS
mlflow-server    mlflow/mlflow   Up (healthy)             0.0.0.0:5000->5000/tcp
api-service      api-service     Up (healthy)             0.0.0.0:8000->8000/tcp
```

---

### 7.4 Akses Layanan

| Layanan | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| API Health Check | http://localhost:8000/health |
| API Dokumentasi (Swagger) | http://localhost:8000/docs |

---

### 7.5 Uji Inferensi

Kirim request prediksi ke API menggunakan `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["i hate all of you", "have a great day everyone"]}'
```

Contoh response:
```json
{
  "predictions": [
    {"text": "i hate all of you", "prediction": 1, "label": "harmful"},
    {"text": "have a great day everyone", "prediction": 0, "label": "neutral"}
  ],
  "model_version": "2",
  "stage": "Production"
}
```

---

### 7.6 Menghentikan Sistem

```bash
# hentikan semua container (data tetap tersimpan di volume)
docker compose down

# hentikan dan hapus semua data volume (reset total)
docker compose down -v
```

## 8. Mengakses Endpoint API

API berjalan di tiga replica pada port `5001`, `5002`, dan `5003`. Setiap replica dapat diakses langsung.

### 8.1 Daftar Endpoint

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/health` | Status API dan model yang dimuat |
| `POST` | `/predict` | Prediksi teks |
| `GET` | `/model/info` | Informasi versi model di registry |

---

### 8.2 Cek Status API

```bash
curl http://localhost:5001/health
```

Contoh respons:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1",
  "stage": "Production"
}
```

---

### 8.3 Prediksi Teks

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I hate you", "have a nice day"]}'
```

Contoh respons:
```json
{
  "predictions": [
    {"text": "I hate you", "prediction": 1, "label": "harmful"},
    {"text": "have a nice day", "prediction": 0, "label": "neutral"}
  ],
  "model_version": "1",
  "stage": "Production"
}
```

---

### 8.4 Test Semua Replica Sekaligus

```bash
for port in 5001 5002 5003; do
  echo "=== Replica port $port ==="
  curl -s -X POST http://localhost:$port/predict \
    -H "Content-Type: application/json" \
    -d '{"texts": ["I hate you"]}' | python3 -m json.tool
done
```

---

## 9. Retraining Trigger dan Ambang Batas Metrik

Sistem mendukung retraining otomatis melalui dua mekanisme, yaitu berdasarkan penurunan performa model (performance-based trigger) dan jadwal berkala (schedule-based trigger).

### 9.1 Performance-Based Trigger

Retraining akan dipicu ketika performa model yang dimonitor melalui Grafana berada di bawah ambang batas yang telah ditentukan. Alert Rule dibuat melalui Grafana UI dan dikonfigurasi untuk mengirimkan Webhook ke GitHub Actions menggunakan `repository_dispatch`.

Contoh ambang batas yang digunakan:

| Metrik   | Threshold |
| -------- | --------- |
| F1 Macro | < 0.75    |

Jika nilai metrik turun di bawah threshold, Grafana akan mengirimkan notifikasi Webhook yang memicu pipeline retraining secara otomatis.

### 9.2 Schedule-Based Trigger

Selain berdasarkan performa, retraining juga dijalankan secara berkala menggunakan GitHub Actions Scheduler.

Konfigurasi yang digunakan:

```yaml
schedule:
  - cron: '0 0 * * 5'
```

Konfigurasi tersebut menjalankan pipeline setiap hari Jumat pukul 00:00 UTC. Mekanisme ini memastikan model tetap diperbarui secara rutin apabila terdapat data baru yang telah ditambahkan ke DVC.

### 9.3 Validasi Model Setelah Retraining

Setelah proses retraining selesai, model baru akan dievaluasi secara otomatis menggunakan metrik yang tersimpan di MLflow.

Model hanya akan didaftarkan ke Model Registry apabila memenuhi seluruh ambang batas berikut:

| Metrik   | Threshold Minimum |
| -------- | ----------------- |
| Accuracy | ≥ 0.85            |
| F1 Macro | ≥ 0.80            |

Jika seluruh syarat terpenuhi, model akan otomatis didaftarkan ke MLflow Model Registry dan dipindahkan ke stage **Staging**. Jika tidak memenuhi threshold, model akan ditolak dan tidak dipromosikan.

---
