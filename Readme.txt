📸🧘‍♂️ Photobooth Pose Yoga Interaktif dengan MediaPipe 🧘‍♀️📸
Repositori ini berisi kode untuk proyek Tugas Besar Pengolahan Citra Digital, yaitu sebuah aplikasi "Photobooth" interaktif yang menggunakan Streamlit dan MediaPipe untuk mendeteksi pose yoga secara real-time. Aplikasi ini akan memandu pengguna untuk melakukan serangkaian pose yoga dan secara otomatis mengambil gambar ketika pose yang dilakukan sudah benar dan stabil.

🌟 Fitur Utama
- Deteksi 6 Pose Yoga: Aplikasi dapat mengenali 6 pose yoga berbeda (plank, tree_pose, downdog, warrior2, half_moon, breath_of_fire) menggunakan model Machine Learning yang telah dilatih.
- Photobooth Otomatis: Memulai countdown 3 detik dan mengambil gambar secara otomatis ketika pose yang benar terdeteksi.
- Antarmuka Interaktif: Dibangun menggunakan Streamlit, memberikan umpan balik visual secara real-time kepada pengguna, termasuk pose target, pose yang terdeteksi, dan tingkat keyakinan model.
- Penghitung Stabilitas: Mencegah pemicu yang tidak disengaja dengan mengharuskan pengguna menahan pose selama beberapa saat.
- Alur Kerja End-to-End: Menyertakan skrip untuk membuat dataset dari gambar, melatih model, dan menjalankan aplikasi utama.

📂 Struktur Repositori
.
├── dataset/                # Folder berisi gambar-gambar untuk melatih model
│   ├── breath_of_fire/
│   ├── cobra/
│   ├── downdog/
│   ├── half_moon/
│   ├── plank/
│   ├── tree_pose/
├── create_dataset.py       # Skrip untuk membuat file CSV dari folder dataset
├── train_model.py          # Skrip untuk melatih model dari file CSV
├── app.py                  # Skrip utama aplikasi photobooth Streamlit
├── yoga_poses.csv          # File dataset fitur yang dihasilkan (opsional)
├── yoga_pose_classifier.pkl # Model machine learning yang sudah dilatih
├── frame.png               # Bingkai untuk photobooth
└── requirements.txt        # Daftar library Python yang dibutuhkan

🛠️ Teknologi yang Digunakan
- Python: Bahasa pemrograman utama.
- Streamlit: Framework untuk membangun antarmuka web interaktif.
- OpenCV: Untuk pemrosesan gambar dan penangkapan video dari webcam.
- MediaPipe Pose: Sebagai ekstraktor fitur untuk mendapatkan 33 landmark tubuh.
- Scikit-learn: Untuk melatih model klasifikasi RandomForestClassifier.
- Pandas & NumPy: Untuk manipulasi data dan operasi numerik.
- Joblib: Untuk menyimpan dan memuat model yang telah dilatih.

🚀 Cara Menjalankan Proyek
1. Persiapan Lingkungan
Pastikan Anda memiliki Python 3.8+ terinstal.

Bash

# Buat dan aktifkan virtual environment (direkomendasikan)
python -m venv venv
source venv/bin/activate  # Untuk Linux/macOS
# venv\Scripts\activate  # Untuk Windows

# Instal semua library yang dibutuhkan
pip install -r requirements.txt

2. Tahap Offline (Jika Ingin Melatih Ulang Model)
Jika Anda ingin membuat dataset dan melatih model dari awal menggunakan gambar Anda sendiri, ikuti langkah ini. Jika tidak, Anda bisa langsung ke Tahap Online.

a. Siapkan Dataset
Buat folder dataset.
Di dalamnya, buat sub-folder untuk setiap pose yang ingin Anda latih (misalnya, plank, tree_pose, dll.).
Isi setiap sub-folder dengan gambar-gambar dari pose yang sesuai.

b. Buat File CSV
Jalankan skrip create_dataset.py untuk memproses gambar di folder dataset menjadi file yoga_poses.csv.

Bash

python create_dataset.py

c. Latih Model
Jalankan skrip train_model.py untuk membaca yoga_poses.csv dan menghasilkan file yoga_pose_classifier.pkl.

Bash

python train_model.py
Outputnya akan menunjukkan akurasi dan laporan klasifikasi model Anda.

3. Tahap Online (Menjalankan Aplikasi Photobooth)
Ini adalah langkah untuk menjalankan aplikasi utama. Pastikan Anda sudah memiliki file yoga_pose_classifier.pkl dan frame.png di dalam folder proyek.

Jalankan perintah berikut di terminal:

Bash

streamlit run app.py
Aplikasi akan secara otomatis terbuka di browser Anda. Ikuti instruksi di layar: pilih kamera, nyalakan, dan mulailah berpose!

👥 Kontributor
Agra (067)
Aulia (070)
Melly (080)
Proyek ini dibuat untuk memenuhi Tugas Besar mata kuliah Pengolahan Citra Digital di Politeknik Negeri Bandung.