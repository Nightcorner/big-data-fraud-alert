# PaySim ML — hasil audit dan eksperimen yang dapat direproduksi

Tanggal eksperimen: 22–24 Oktober 2026  
Status: pipeline scikit-learn/LightGBM dan Spark MLlib sudah dijalankan di lingkungan Colab. Pada versi kode yang disiapkan untuk penyerahan, definisi skema Spark telah dirapikan agar memetakan seluruh 11 kolom CSV sesuai urutan aslinya.

## 1. Audit dataset

File: `data/PS_20174392719_1491204439457_log.csv`

- Total: 6.362.620 transaksi, 11 kolom.
- Label `isFraud=1`: 8.213 transaksi.
- Label `isFraud=0`: 6.354.407 transaksi.
- Missing values: 0 untuk semua kolom.
- Fraud pada dataset: sekitar 0,129% dari seluruh baris.
- Fraud hanya muncul pada jenis `TRANSFER` dan `CASH_OUT` pada dataset ini.
- `isFlaggedFraud=1` muncul pada 16 baris dan semua 16 mempunyai `isFraud=1`. Kolom ini dikecualikan dari ML karena merupakan flag dari detektor lain.

Temuan pola simulator: pada 8.034 dari 8.213 transaksi fraud, `amount` sama persis dengan `oldbalanceOrg`; tidak ditemukan baris non-fraud dengan kesamaan persis tersebut. Ini merupakan sinyal yang sangat kuat dan bersifat spesifik terhadap mekanisme pembangkitan PaySim. Ini **bukan bukti bahwa pola ini berlaku pada transaksi bank sungguhan**.

## 2. Kebijakan fitur

Input model tidak mencakup:

- `isFraud`: target, tidak boleh menjadi fitur.
- `isFlaggedFraud`: flag deteksi yang sudah tersedia, dikecualikan untuk menghindari model hanya meniru sistem lain.
- `newbalanceOrig`, `newbalanceDest`: saldo sesudah transaksi, dikecualikan karena berpotensi tidak tersedia pada saat alert harus dibuat.
- `nameOrig`, `nameDest`: ID akun persis, dikecualikan dari model utama untuk mengurangi hafalan ID ber-cardinality tinggi.

Eksperimen fitur inti memakai `step`, `type`, dan `amount`. Eksperimen fitur saldo menambah `oldbalanceOrg`, `oldbalanceDest`, dan fitur turunan dari data pre-event. `amount_equals_oldbalanceOrg` dibuat dari dua kolom transaksi sebelum hasil akhir dihitung; fitur tersebut tetap harus ditafsirkan sebagai pola simulator.

## 3. Pembagian waktu

Pembagian tidak acak agar data masa depan tidak masuk ke training:

| Partisi | Step | Baris | Fraud | Non-fraud |
|---|---:|---:|---:|---:|
| Train | 1–520 | 6.082.007 | 5.781 | 6.076.226 |
| Validation | 521–600 | 177.040 | 832 | 176.208 |
| Test | 601–743 | 103.573 | 1.600 | 101.973 |

Tingkat fraud berubah cukup besar antara rentang waktu ini. Karena itu, hasil test mewakili bagian akhir simulator, bukan random test set dan bukan data bank nyata.

## 4. Hasil validasi fitur dan model

Model menggunakan LightGBM melalui API scikit-learn. Kedua kandidat dilatih dengan training sampai step 520, lalu dibandingkan pada validation. Kandidat dipilih menggunakan Average Precision validation; threshold klasifikasi dipilih menggunakan F1 validation.

| Kandidat | Validation ROC-AUC | Validation Average Precision | Threshold F1 validation |
|---|---:|---:|---:|
| Fitur inti: step, type, amount | 0,8881 | 0,2300 | 0,25160 |
| Fitur saldo pre-event | 1,0000 | 1,0000 | 0,005471 |

Pada model fitur saldo, LightGBM menghentikan proses pada iterasi terbaik ke-1 karena validation terpisah sempurna. Feature importance menempatkan `amount_equals_oldbalanceOrg` jauh di atas fitur lain. Ini tanda kuat bahwa skor nyaris sempurna terutama diperoleh dari satu pola sangat khas simulator, bukan bukti bahwa metode umum akan mencapai skor yang sama pada fraud perbankan nyata.

## 5. Hasil pada temporal test

Threshold dipilih dari validation, bukan dari label test. Hasil model fitur saldo pada step 601–743:

| Metrik | Hasil aktual |
|---|---:|
| Total transaksi test | 103.573 |
| Fraud sebenarnya (`isFraud=1`) | 1.600 |
| TP — fraud ditemukan | 1.599 |
| FP — non-fraud salah ditandai | 0 |
| FN — fraud terlewat | 1 |
| TN — non-fraud dikenal dengan benar | 101.973 |
| Total alert di threshold validation | 1.599 |
| Precision | 100% |
| Recall | 99,9375% |
| F1 | 99,9687% |
| ROC-AUC | 0,999686 |
| Average Precision | 0,999385 |

Dengan threshold default 0,5, model memberi 1.550 TP, 0 FP, dan 50 FN. Threshold validation yang lebih rendah menangkap 49 kasus tambahan tanpa FP pada temporal test ini.

Untuk kandidat fitur inti tanpa saldo, test ROC-AUC 0,8873 dan Average Precision 0,3248. Pada threshold yang dipilih dari validation, hasilnya 323 TP, 34 FP, 1.277 FN, dan 101.939 TN. Ini menunjukkan bahwa fitur saldo dan pola `amount == oldbalanceOrg` memberikan sebagian besar peningkatan performa.

## 6. Interpretasi yang diperbolehkan dan yang tidak

Boleh dilaporkan:

- Pada temporal holdout PaySim yang didefinisikan, model dengan fitur pre-event saldo mendapat TP 1.599, FP 0, FN 1.
- Dataset sintetis ini mempunyai hubungan yang sangat kuat antara `amount` dan `oldbalanceOrg` pada kelas fraud.
- Model lebih berhasil mempelajari pola generator PaySim daripada model fitur inti.

Tidak boleh disimpulkan:

- Model mendeteksi 99,94% fraud di bank nyata.
- Model ini sudah terbukti untuk EDC/Base24, tarik tunai luar negeri, atau tiga business rules bank.
- Semua jenis fraud di dunia bisa dideteksi dengan precision 100%.
- Spark MLlib mempunyai performa yang sama sebelum skrip Spark berhasil dijalankan.

Kualitas eksperimen baik jika metode dan keterbatasan dilaporkan dengan jujur; skor tinggi sendiri bukan bukti validitas eksternal.

## 7. File hasil utama

- `outputs/audit/paysim_audit.json`: audit seluruh dataset.
- `outputs/sklearn/validation_candidate_comparison.csv`: perbandingan yang dipilih pada validation.
- `outputs/sklearn/test_feature_ablation_comparison.csv`: audit dampak fitur pada temporal test.
- `outputs/sklearn/final_test_metrics.json`: hasil utama.
- `outputs/sklearn/confusion_matrix.csv`: TP/FP/FN/TN.
- `outputs/sklearn/test_alert_budget_metrics.csv`: trade-off pada batas jumlah alert.
- `outputs/sklearn/feature_importance.csv`: nama fitur dan importance.
- `outputs/graph/graph_summary.json`: graf akun pada sampel sistematis 1 dari setiap 64 transaksi.

## 8. Keterbatasan lain

- PaySim adalah dataset sintetis; pola fraud disusun oleh simulator.
- Test temporal tetap berasal dari dataset yang sama dan hanya mewakili rentang step terakhir.
- Eksperimen Spark MLlib telah dijalankan di Google Colab pada 24 Oktober 2026. Source code dalam paket v2 merapikan definisi skema menjadi 11 kolom penuh; jalankan ulang notebook terlampir apabila ingin mereproduksi hasil dari file source yang sudah diperbaiki. Angka di file hasil tetap harus diperlakukan sebagai hasil pada PaySim sintetis.
- Graph NetworkX berjalan pada sampel sekitar 99 ribu transaksi karena keterbatasan memori; ini analisis deskriptif sampel, bukan pembuktian skalabilitas graf penuh.
- Business rules Proposed v2 tetap terpisah, menggunakan fixture simulasi mereka sendiri, dan checksum konfigurasi tidak berubah.
