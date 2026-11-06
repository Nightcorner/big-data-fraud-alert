# ASG2 Big Data — Source Code dan Artefak Eksperimen PaySim

Paket ini berisi kode sumber yang digunakan untuk eksperimen ASG2, konfigurasi business rules, unit tests, ringkasan hasil LightGBM dan graph analytics, serta artefak hasil eksekusi Spark MLlib.

## Komponen

- `run_paysim_audit.py`: audit struktur dan kualitas dataset PaySim.
- `run_paysim_sklearn.py`: eksperimen LightGBM dengan API scikit-learn.
- `run_paysim_spark.py`: eksperimen Random Forest menggunakan Spark MLlib.
- `run_paysim_graph.py`: graph analytics menggunakan NetworkX pada sampel sistematis 1 dari 64 transaksi.
- `src/`: fungsi feature engineering dan business rules.
- `config/` dan `tests/`: konfigurasi aturan serta pengujian.
- `outputs/sklearn/`: metrik, prediksi, dan hasil eksperimen LightGBM.
- `outputs/graph/`: ringkasan graf dan hasil agregasi sampel.
- `outputs/spark/`: log eksekusi Colab, metrik evaluasi, prediksi test, feature importance, environment, dan model Spark MLlib yang dihasilkan dari eksekusi.
- `../notebooks/Eksperimen_Spark_MLlib_PaySim_v2_11kolom.ipynb` adalah notebook sumber yang dapat dijalankan ulang. Output run sebelumnya disimpan terpisah pada `outputs/spark/`; sel notebook sengaja tidak menyimpan output run lama agar tidak mencampur hasil eksekusi.
- `../notebooks/Eksperimen_Spark_MLlib_PaySim_v2_11kolom.ipynb`: notebook Colab untuk menjalankan ulang eksperimen Spark MLlib.

## Dataset

File CSV mentah tidak disertakan di dalam ZIP ini. Dataset yang digunakan adalah PaySim `PS_20174392719_1491204439457_log.csv`, tersedia pada Kaggle:

https://www.kaggle.com/datasets/ealaxi/paysim1

Setelah mengunduh dan mengekstrak dataset, letakkan file CSV pada:

`data/PS_20174392719_1491204439457_log.csv`

Jangan mengubah urutan atau nama 11 kolom aslinya. Tautan dataset juga dicantumkan terpisah pada deliverables laporan.

## Lingkungan dan menjalankan kode

Untuk eksperimen LightGBM dan graph analytics, gunakan Python 3.10–3.12. Dari direktori `PaySim_Fraud_Alert_Repaired_v2`:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_rules_freeze.py
python run_paysim_audit.py
python run_paysim_sklearn.py
python run_paysim_graph.py
```

Eksperimen Spark MLlib memerlukan Java 17+ dan PySpark 4.0.x. Cara menjalankannya melalui Google Colab tersedia pada notebook di folder `notebooks`. Skrip utama yang dipanggil notebook adalah `run_paysim_spark.py`.

## Catatan interpretasi dan batasan

1. PaySim merupakan dataset sintetis mobile-money, bukan catatan transaksi tarik tunai luar negeri melalui EDC/Base24.
2. Dataset PaySim tidak memuat semua atribut yang diperlukan untuk menguji business rules bank. Karena itu, business rules diuji terpisah menggunakan skenario sintetis.
3. Analisis graf memakai sampel sistematis 1 dari 64 baris, bukan seluruh dataset.
4. Eksperimen Spark dijalankan pada lingkungan Colab/local Spark; hasil ini tidak membuktikan skalabilitas multi-node.
5. Tingginya skor model dengan fitur saldo dipengaruhi pola khas simulator PaySim dan tidak boleh dianggap sebagai bukti performa setara pada transaksi bank nyata.

Kode dan hasil dalam paket ini harus dibaca bersama laporan ASG2, yang menjelaskan metode, hasil, serta batasan eksperimennya.
