import os
import cv2
import mediapipe as mp
import csv
import numpy as np

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Path ke direktori dataset Anda
DATASET_PATH = "dataset"
# Nama file CSV output
CSV_FILE = "yoga_poses.csv"

# Siapkan header untuk file CSV
# Header akan berisi 'label' dan 33 landmark dengan 4 koordinat (x, y, z, visibility)
# Total kolom = 1 (label) + 33 * 4 = 133
landmarks_header = []
for i in range(33):
    landmarks_header += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']
csv_header = ['label'] + landmarks_header

# Mulai menulis ke file CSV
with open(CSV_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

    # Loop melalui setiap sub-folder (setiap pose)
    for pose_label in os.listdir(DATASET_PATH):
        pose_path = os.path.join(DATASET_PATH, pose_label)
        if not os.path.isdir(pose_path):
            continue

        # Loop melalui setiap gambar dalam folder pose
        for image_name in os.listdir(pose_path):
            image_path = os.path.join(pose_path, image_name)
            
            # Baca gambar
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Gagal membaca gambar: {image_path}")
                    continue
                
                # Konversi ke RGB dan proses
                results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                # Jika landmark terdeteksi
                if results.pose_landmarks:
                    # Ekstrak landmark dan ratakan (flatten) menjadi satu baris
                    landmarks = results.pose_landmarks.landmark
                    row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten())
                    
                    # Tambahkan label di awal baris
                    row.insert(0, pose_label)
                    
                    # Tulis baris ke file CSV
                    writer.writerow(row)
                else:
                    print(f"Tidak ada pose terdeteksi di: {image_path}")

            except Exception as e:
                print(f"Error memproses {image_path}: {e}")

print(f"Dataset berhasil dibuat dan disimpan di {CSV_FILE}")