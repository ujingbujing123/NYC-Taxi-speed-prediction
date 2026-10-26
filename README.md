# NYC Taxi: Kecepatan & Kemacetan (Streamlit)

Isi folder yang harus ada berdampingan:

```
app.py  Dockerfile  requirements.txt
speed_model.joblib  strict_model.joblib  meta.joblib  app_data.joblib
```

`requirements.txt` dan keempat file `.joblib` dihasilkan oleh sel ekspor di notebook (unduh dari panel Files Colab).
Versi scikit-learn/pandas/numpy di `requirements.txt` sengaja sama dengan versi di Colab agar model bisa dibaca.

## Jalankan lokal
```
pip install -r requirements.txt
streamlit run app.py
```

## Jalankan dengan Docker
```
docker build -t nyc-taxi-speed .
docker run -p 8501:8501 nyc-taxi-speed
```
Buka http://localhost:8501

## Deploy gratis
Unggah semua file di atas ke repositori GitHub, lalu deploy di Streamlit Community Cloud dengan `app.py` sebagai file utama.

## Monitoring dan pembaruan model (untuk slide MLOps)
- Latih ulang berkala (mis. bulanan) saat data baru masuk; bandingkan MAE/bias pada data terbaru dengan nilai di tab Insight.
- Pantau bias: kecepatan turun di akhir 2025, jadi model rute cenderung memprediksi terlalu cepat.
- Simpan versi model dan versi library bersama artefak.
