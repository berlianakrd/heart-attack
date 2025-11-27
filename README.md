# Indonesia Heart Attack Prediction

Sistem prediksi risiko serangan jantung berbasis Machine Learning untuk populasi Indonesia.

## 📋 Deskripsi Project

Project ini merupakan implementasi Data Science Life Cycle untuk memprediksi risiko serangan jantung pada populasi Indonesia menggunakan berbagai faktor risiko seperti demografi, klinis, gaya hidup, dan lingkungan.

## 🎯 Tujuan

- Memprediksi risiko serangan jantung berdasarkan data kesehatan individu
- Memberikan analisis faktor-faktor risiko utama
- Membantu deteksi dini untuk pencegahan serangan jantung

## 📊 Dataset

Dataset terdiri dari 500 sampel data kesehatan individu di Indonesia dengan 28 fitur:
- **Demographics**: age, gender, region, income_level
- **Clinical Risk Factors**: hypertension, diabetes, cholesterol_level, obesity, dll
- **Lifestyle Factors**: smoking_status, alcohol_consumption, physical_activity, dll
- **Environmental Factors**: air_pollution_exposure, stress_level, sleep_hours
- **Medical Screening**: blood_pressure, fasting_blood_sugar, EKG_results, dll
- **Target**: heart_attack (0=No, 1=Yes)

## 🔄 Data Science Life Cycle

Project ini mengikuti 7 tahapan Data Science Life Cycle:

1. **Business Understanding**: Memahami masalah serangan jantung di Indonesia
2. **Data Mining**: Mengumpulkan dan mengekstrak data
3. **Data Cleaning**: Membersihkan data dari missing values dan inkonsistensi
4. **Data Exploration**: Analisis eksplorasi data (EDA)
5. **Feature Engineering**: Seleksi dan pembuatan fitur baru
6. **Predictive Modeling**: Training model klasifikasi (Logistic Regression, Decision Tree, KNN)
7. **Data Visualization**: Visualisasi hasil dan insights

## 🛠️ Teknologi yang Digunakan

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Frontend**: HTML, CSS, JavaScript

## 📁 Struktur Project
```
indonesia-heart-attack-prediction/
├── data/                      # Dataset
├── notebooks/                 # Jupyter notebooks untuk setiap tahap
├── src/                       # Source code
├── models/                    # Saved models
├── static/                    # CSS & JS
├── templates/                 # HTML templates
├── app.py                     # Flask application
└── requirements.txt           # Dependencies
```

## 🚀 Cara Menjalankan

1. **Clone repository**
```bash
git clone [repository-url]
cd indonesia-heart-attack-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Jalankan Jupyter Notebooks**
```bash
jupyter notebook
```
Buka dan jalankan notebooks di folder `notebooks/` secara berurutan (01-07)

4. **Jalankan Web Application**
```bash
python app.py
```
Akses di browser: `http://localhost:5000`

## 📈 Model Performance

Model yang digunakan:
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)

Metrik evaluasi: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## 👥 Fitur Web Application

1. **Home**: Informasi umum tentang serangan jantung di Indonesia
2. **Prediction**: Form input untuk prediksi risiko serangan jantung
3. **Analysis**: Visualisasi dan analisis data
4. **About**: Informasi tentang project dan metodologi

## 📝 Catatan

Project ini dibuat untuk keperluan akademik mata kuliah Artificial Intelligence dengan fokus pada Supervised Learning - Classification.

## 📧 Kontak

[Nama Anda]
[Email/Kontak]

---
© 2024 Indonesia Heart Attack Prediction