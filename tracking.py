import cv2
import mediapipe as mp
import numpy as np
from collections import deque # <-- BARU: Import library deque

# --- FUNGSI UNTUK MENEMPELKAN GAMBAR PNG (DENGAN TRANSPARANSI) ---
# Fungsi ini sudah bagus, tidak perlu diubah.
def overlay_png(background, overlay, x, y):
    """
    Menempelkan gambar 'overlay' (dengan alpha channel) ke 'background'.
    x, y adalah koordinat pojok kiri atas dari gambar overlay.
    """
    bg_h, bg_w, _ = background.shape
    overlay_h, overlay_w, _ = overlay.shape

    if x >= bg_w or y >= bg_h or x + overlay_w <= 0 or y + overlay_h <= 0:
        return background

    y1, y2 = max(0, y), min(bg_h, y + overlay_h)
    x1, x2 = max(0, x), min(bg_w, x + overlay_w)

    alpha_h_start = max(0, -y)
    alpha_h_end = y2 - y1 + alpha_h_start
    alpha_w_start = max(0, -x)
    alpha_w_end = x2 - x1 + alpha_w_start
    
    roi = background[y1:y2, x1:x2]
    overlay_area = overlay[alpha_h_start:alpha_h_end, alpha_w_start:alpha_w_end]

    if roi.shape[0] != overlay_area.shape[0] or roi.shape[1] != overlay_area.shape[1]:
        return background

    alpha = overlay_area[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        roi[:, :, c] = (alpha * overlay_area[:, :, c] + alpha_inv * roi[:, :, c])
    
    background[y1:y2, x1:x2] = roi
    return background

# --- INISIALISASI MEDIAPIPE ---
mp_holistic = mp.solutions.holistic

# --- MUAT SEMUA GAMBAR KARAKTER DAN SKALAKAN ---
try:
    img_head_normal = cv2.imread('margo_head_normal.png', cv2.IMREAD_UNCHANGED)
    img_body_normal = cv2.imread('margo_body_normal.png', cv2.IMREAD_UNCHANGED)
    img_head_react = cv2.imread('margo_head_react.png', cv2.IMREAD_UNCHANGED)
    img_body_react = cv2.imread('margo_body_react.png', cv2.IMREAD_UNCHANGED)
    
    if any(img is None for img in [img_head_normal, img_body_normal, img_head_react, img_body_react]):
        raise FileNotFoundError("Salah satu atau lebih file gambar tidak ditemukan.")
    
    # --- PENTING: FAKTOR SKALA ---
    scale_factor_head = 0.45
    scale_factor_body = 0.7 

    img_head_normal = cv2.resize(img_head_normal, (0,0), fx=scale_factor_head, fy=scale_factor_head, interpolation=cv2.INTER_AREA)
    img_body_normal = cv2.resize(img_body_normal, (0,0), fx=scale_factor_body, fy=scale_factor_body, interpolation=cv2.INTER_AREA)
    img_head_react = cv2.resize(img_head_react, (0,0), fx=scale_factor_head, fy=scale_factor_head, interpolation=cv2.INTER_AREA)
    img_body_react = cv2.resize(img_body_react, (0,0), fx=scale_factor_body, fy=scale_factor_body, interpolation=cv2.INTER_AREA)

except FileNotFoundError as e:
    print(e)
    print("Pastikan semua file gambar ('margo_head_normal.png', 'margo_body_normal.png', dll) ada di folder yang sama dengan script Python.")
    exit()

# --- BUKA KAMERA & INISIALISASI MODEL HOLISTIC ---
cap = cv2.VideoCapture(0)

# <-- BARU: BUAT PENYIMPANAN HISTORY UNTUK SMOOTHING -->
# maxlen=5 berarti kita akan menyimpan & merata-ratakan 5 posisi terakhir
history_head = deque(maxlen=5)
history_body = deque(maxlen=5)

with mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        # --- LOGIKA UTAMA ---
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            
            # 1. Tentukan status (Normal atau Bereaksi)
            left_wrist = pose[mp_holistic.PoseLandmark.LEFT_WRIST]
            right_wrist = pose[mp_holistic.PoseLandmark.RIGHT_WRIST]
            nose = pose[mp_holistic.PoseLandmark.NOSE]
            is_reacting = left_wrist.y < nose.y and right_wrist.y < nose.y

            # 2. Pilih aset gambar yang sesuai dengan status
            if is_reacting:
                current_head_img = img_head_react
                current_body_img = img_body_react
            else:
                current_head_img = img_head_normal
                current_body_img = img_body_normal
                
            current_head_h, current_head_w, _ = current_head_img.shape
            current_body_h, current_body_w, _ = current_body_img.shape

            # --------------------------------------------------------------------
            # ## PERHITUNGAN POSISI YANG SUDAH DIHALUSKAN (SMOOTHED) ##
            
            # 3.1 Dapatkan posisi mentah (RAW) dari landmark
            raw_pos_head = (int(nose.x * frame_w), int(nose.y * frame_h))

            left_shoulder = pose[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            raw_pos_body = (
                int((left_shoulder.x + right_shoulder.x) * frame_w / 2),
                int((left_shoulder.y + right_shoulder.y) * frame_h / 2)
            )

            # 3.2 Simpan posisi mentah ke dalam history
            history_head.append(raw_pos_head)
            history_body.append(raw_pos_body)

            # 3.3 Hitung posisi RATA-RATA dari history
            avg_x_head = sum(p[0] for p in history_head) // len(history_head)
            avg_y_head = sum(p[1] for p in history_head) // len(history_head)
            avg_x_body = sum(p[0] for p in history_body) // len(history_body)
            avg_y_body = sum(p[1] for p in history_body) // len(history_body)

            # 3.4 Gunakan posisi RATA-RATA ini untuk menggambar
            cx_head, cy_head = avg_x_head, avg_y_head
            cx_shoulder_mid, cy_shoulder_mid = avg_x_body, avg_y_body
            # --------------------------------------------------------------------
            
            # ======== SESUAIKAN OFFSET INI UNTUK POSISI YANG PAS ========
            offset_y_body = 180  # Geser badan ke atas/bawah
            offset_y_head = -10  # Geser kepala ke atas/bawah agar pas dengan leher
            # ==========================================================
            
            pos_x_body = cx_shoulder_mid - (current_body_w // 2)
            pos_y_body = cy_shoulder_mid - (current_body_h // 2) + offset_y_body

            pos_x_head = cx_head - (current_head_w // 2)
            pos_y_head = cy_head - (current_head_h // 2) + offset_y_head

            # 4. Gambar Karakter ke Layar
            frame = overlay_png(frame, current_body_img, pos_x_body, pos_y_body)
            frame = overlay_png(frame, current_head_img, pos_x_head, pos_y_head)
            
        cv2.imshow('VTuber Margo Lengkap', frame)

        if cv2.waitKey(5) & 0xFF in [ord('q'), 27]: # Keluar dengan 'q' atau 'ESC'
            break

cap.release()
cv2.destroyAllWindows()