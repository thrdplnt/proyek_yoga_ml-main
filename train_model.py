import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # Untuk menyimpan model

# 1. Muat Dataset
df = pd.read_csv("yoga_poses.csv")

# 2. Siapkan Data
# X adalah fitur (semua kolom kecuali 'label')
# y adalah target (kolom 'label')
X = df.drop('label', axis=1)
y = df['label']

# 3. Bagi Data menjadi Data Latih dan Data Uji
# 80% untuk latih, 20% untuk uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Latih Model
# Kita gunakan RandomForestClassifier, sama seperti di proyek referensi Anda
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model: {accuracy * 100:.2f}%")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# 6. Simpan Model yang Sudah Dilatih
model_filename = "yoga_pose_classifier.pkl"
joblib.dump(model, model_filename)
print(f"\nModel berhasil dilatih dan disimpan sebagai {model_filename}")