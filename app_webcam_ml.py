import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw
import io
import time
import mediapipe as mp
import joblib
import atexit

# --- Konfigurasi Aplikasi ---
POSE_SEQUENCE = [
    "plank", "tree_pose", "downdog", "cobra", "half_moon", "breath_of_fire"
]
MAX_PHOTOS = len(POSE_SEQUENCE)
CONFIDENCE_THRESHOLD = 0.64

# --- Inisialisasi Model & MediaPipe ---
st.set_page_config(layout="wide")
st.title("🧘‍♀️📸 Yoga Pose Classifier Photobooth")

try:
    model = joblib.load("yoga_pose_classifier.pkl")
except FileNotFoundError:
    st.error("FATAL ERROR: File 'yoga_pose_classifier.pkl' tidak ditemukan.")
    st.warning("Pastikan file model (.pkl) berada di folder yang sama.")
    st.stop()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Fungsi Bantuan ---
def release_camera():
    if 'cap' in st.session_state and st.session_state.get('cap') and st.session_state.cap.isOpened():
        st.session_state.cap.release()
    if 'pose' in globals():
        pose.close()
atexit.register(release_camera)

# <<< BARU: Fungsi untuk mencari kamera yang tersedia >>>
def get_available_camera_indices(max_to_check=5):
    indices = []
    for i in range(max_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            indices.append(i)
            cap.release()
    return indices

def crop_and_resize_image(pil_image, target_w, target_h):
    img_width, img_height = pil_image.size; target_ratio = target_w / target_h; current_ratio = img_width / img_height
    if current_ratio > target_ratio: new_width = int(img_height * target_ratio); left = (img_width - new_width) // 2; pil_image = pil_image.crop((left, 0, left + new_width, img_height))
    else: new_height = int(img_width / target_ratio); top = (img_height - new_height) // 2; pil_image = pil_image.crop((0, top, img_width, top + new_height))
    return pil_image.resize((target_w, target_h), Image.LANCZOS)

def create_composite_image(base_frame_pil, live_cv2_frame=None, active_slot_idx=0, captured_photos_pil=None, slot_definitions=None):
    if base_frame_pil is None: return None
    background = Image.new('RGBA', base_frame_pil.size, (255, 255, 255, 0))
    if captured_photos_pil:
        for i, photo_pil in enumerate(captured_photos_pil):
            if photo_pil: x, y, w, h = slot_definitions[i]; background.paste(photo_pil, (x, y))
    if live_cv2_frame is not None and 0 <= active_slot_idx < len(slot_definitions):
        x_active, y_active, w_active, h_active = slot_definitions[active_slot_idx]
        live_pil_rgb = Image.fromarray(cv2.cvtColor(live_cv2_frame, cv2.COLOR_BGR2RGB))
        resized_live_pil = crop_and_resize_image(live_pil_rgb, w_active, h_active); background.paste(resized_live_pil, (x_active, y_active))
    composite = Image.alpha_composite(background, base_frame_pil)
    if live_cv2_frame is not None and 0 <= active_slot_idx < len(slot_definitions):
        x, y, w, h = slot_definitions[active_slot_idx]; draw = ImageDraw.Draw(composite); draw.rectangle([x-5, y-5, x+w+5, y+h+5], outline="lime", width=5)
    return composite

# --- Inisialisasi State Aplikasi ---
if 'camera_active' not in st.session_state: st.session_state.camera_active = False
if 'active_slot' not in st.session_state: st.session_state.active_slot = 0
if 'captured_photos' not in st.session_state: st.session_state.captured_photos = [None] * MAX_PHOTOS
if 'photobooth_complete' not in st.session_state: st.session_state.photobooth_complete = False
if 'countdown' not in st.session_state: st.session_state.countdown = -1
if 'countdown_time' not in st.session_state: st.session_state.countdown_time = 0
if 'camera_indices' not in st.session_state: st.session_state.camera_indices = get_available_camera_indices()

# --- Tampilan Utama & Kontrol ---
FRAME_PATH = "frame.png"; photobooth_frame_pil = Image.open(FRAME_PATH).convert("RGBA")
SLOT_WIDTH, SLOT_HEIGHT = 500, 400; X_COL1, Y_ROW1 = 55, 120; Y_ROW2, Y_ROW3 = 535, 1000; X_COL2 = 655
slot_definitions = [(X_COL1, Y_ROW1, SLOT_WIDTH, SLOT_HEIGHT), (X_COL1, Y_ROW2, SLOT_WIDTH, SLOT_HEIGHT), (X_COL1, Y_ROW3, SLOT_WIDTH, SLOT_HEIGHT), (X_COL2, Y_ROW1, SLOT_WIDTH, SLOT_HEIGHT), (X_COL2, Y_ROW2, SLOT_WIDTH, SLOT_HEIGHT), (X_COL2, Y_ROW3, SLOT_WIDTH, SLOT_HEIGHT)][:MAX_PHOTOS]

# <<< BARU: Dropdown untuk memilih kamera >>>
if not st.session_state.camera_active:
    if not st.session_state.camera_indices:
        st.error("Tidak ada kamera yang terdeteksi.")
    else:
        st.session_state.selected_camera = st.selectbox(
            "Pilih Kamera Anda:",
            options=st.session_state.camera_indices,
            format_func=lambda x: f"Kamera {x}"
        )

col1, col2 = st.columns(2)
with col1:
    if st.button('🔌 Nyalakan/Matikan Kamera', type="primary", use_container_width=True, disabled=not st.session_state.camera_indices):
        st.session_state.camera_active = not st.session_state.camera_active
        if not st.session_state.camera_active: release_camera()
        st.rerun() # <<< BARU: Paksa Rerun untuk memastikan state diperbarui
with col2:
    if st.button("🔄 Ulangi Sesi", use_container_width=True):
        st.session_state.camera_active = False; st.session_state.active_slot = 0; st.session_state.captured_photos = [None] * MAX_PHOTOS; st.session_state.photobooth_complete = False; st.session_state.countdown = -1; release_camera(); st.rerun()

if not st.session_state.photobooth_complete:
    target_pose_name = POSE_SEQUENCE[st.session_state.active_slot]
    st.header(f"SLOT {st.session_state.active_slot + 1}/{MAX_PHOTOS}: Lakukan Pose '{target_pose_name.upper()}'")
else:
    st.header("🎉 Sesi Selesai! 🎉")
live_preview_placeholder = st.empty()

# --- Logika Inti Aplikasi ---
if st.session_state.camera_active and not st.session_state.photobooth_complete:
    # <<< BARU: Gunakan kamera yang dipilih dari dropdown >>>
    selected_cam_idx = st.session_state.get('selected_camera', 0)
    if 'cap' not in st.session_state or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(selected_cam_idx)
    
    cap = st.session_state.cap
    if not cap.isOpened():
        st.error(f"Gagal membuka Kamera {selected_cam_idx}. Coba pilih indeks lain.")
    else:
        while cap.isOpened() and st.session_state.camera_active:
            # ... (Sisa logika di dalam while loop tetap sama persis) ...
            ret, raw_frame = cap.read();
            if not ret: break
            display_frame = cv2.flip(raw_frame.copy(), 1)
            image_for_detection = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_for_detection)
            detected_pose, proba = "UNKNOWN", 0.0
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                try:
                    landmarks = results.pose_landmarks.landmark
                    row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten())
                    X = np.array(row).reshape(1, -1)
                    detected_pose = model.predict(X)[0]
                    proba = max(model.predict_proba(X)[0])
                    if (detected_pose.lower() == target_pose_name.lower() and proba > CONFIDENCE_THRESHOLD and st.session_state.countdown == -1):
                        st.session_state.countdown = 3; st.session_state.countdown_time = time.time()
                except Exception as e: pass
            cv2.rectangle(display_frame, (0, 0), (550, 100), (20, 20, 20), -1)
            cv2.putText(display_frame, f"TARGET: {target_pose_name.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            color_detected = (0, 255, 0) if detected_pose.lower() == target_pose_name.lower() else (255, 255, 255)
            cv2.putText(display_frame, f"TERDETEKSI: {detected_pose.upper()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_detected, 2)
            color_proba = (0, 255, 0) if proba > CONFIDENCE_THRESHOLD else (255, 255, 255)
            cv2.putText(display_frame, f"KEYAKINAN: {int(proba*100)}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_proba, 2)
            if st.session_state.countdown > -1:
                if st.session_state.countdown > 0: text = str(st.session_state.countdown); cv2.putText(display_frame, text, (int(display_frame.shape[1]/2)-50, int(display_frame.shape[0]/2)+50), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 25)
                if time.time() - st.session_state.countdown_time >= 1: st.session_state.countdown -= 1; st.session_state.countdown_time = time.time()
                if st.session_state.countdown == 0:
                    st.toast(f"Pose '{target_pose_name}' Berhasil!"); active_slot = st.session_state.active_slot
                    clean_frame_for_photo = cv2.flip(raw_frame, 1); pil_cap_rgb = Image.fromarray(cv2.cvtColor(clean_frame_for_photo, cv2.COLOR_BGR2RGB))
                    x_slot, y_slot, w_slot, h_slot = slot_definitions[active_slot]; st.session_state.captured_photos[active_slot] = crop_and_resize_image(pil_cap_rgb, w_slot, h_slot)
                    st.session_state.active_slot += 1; st.session_state.countdown = -1
                    if st.session_state.active_slot >= MAX_PHOTOS: st.session_state.photobooth_complete = True; st.session_state.camera_active = False
                    st.rerun()
            composite_img = create_composite_image(photobooth_frame_pil, display_frame, st.session_state.active_slot, st.session_state.captured_photos, slot_definitions)
            live_preview_placeholder.image(composite_img, use_container_width=True)
else:
    final_image = create_composite_image(photobooth_frame_pil, captured_photos_pil=st.session_state.captured_photos, slot_definitions=slot_definitions)
    if final_image: live_preview_placeholder.image(final_image, use_container_width=True)

if st.session_state.photobooth_complete:
    st.balloons(); buf = io.BytesIO(); final_image.save(buf, format="PNG"); st.download_button("📥 Unduh Hasil Photostrip", buf.getvalue(), "photostrip_yoga.png", "image/png", use_container_width=True)