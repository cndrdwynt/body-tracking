import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

# --- FUNGSI UNTUK MENEMPELKAN GAMBAR PNG (DENGAN TRANSPARANSI) ---
def overlay_png(background, overlay, x, y):
    # ... (Fungsi ini tidak berubah)
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
mp_hands = mp.solutions.hands

# --- MUAT SEMUA GAMBAR KARAKTER DAN SKALAKAN ---
try:
    # ... (Kode muat aset Margo tidak berubah)
    img_head_normal = cv2.imread('margo_head_normal.png', cv2.IMREAD_UNCHANGED)
    img_head_blush = cv2.imread('margo_head_blush.png', cv2.IMREAD_UNCHANGED)
    img_head_excited = cv2.imread('margo_head_excited.png', cv2.IMREAD_UNCHANGED) 
    img_body_normal = cv2.imread('margo_body_normal.png', cv2.IMREAD_UNCHANGED)
    img_body_coverface = cv2.imread('margo_body_coverface.png', cv2.IMREAD_UNCHANGED)
    img_body_peace = cv2.imread('margo_body_peace.png', cv2.IMREAD_UNCHANGED)
    img_body_ok = cv2.imread('margo_body_ok.png', cv2.IMREAD_UNCHANGED)
    img_body_wave = cv2.imread('margo_body_wave.png', cv2.IMREAD_UNCHANGED)
    img_body_excited = cv2.imread('margo_body_excited.png', cv2.IMREAD_UNCHANGED)
    img_body_thumbsup = cv2.imread('margo_body_thumbsup.png', cv2.IMREAD_UNCHANGED)
    
    image_assets = [
        img_head_normal, img_head_blush, img_head_excited, img_body_normal, img_body_coverface,
        img_body_peace, img_body_ok, img_body_wave, img_body_excited, img_body_thumbsup
    ]
    if any(img is None for img in image_assets):
        raise FileNotFoundError("Salah satu atau lebih file gambar tidak ditemukan! Cek kembali semua 10 nama file.")
    
    scale_factor_head = 0.4
    scale_factor_body = 0.5 

    img_head_normal = cv2.resize(img_head_normal, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_blush = cv2.resize(img_head_blush, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_excited = cv2.resize(img_head_excited, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_body_normal = cv2.resize(img_body_normal, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_coverface = cv2.resize(img_body_coverface, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_peace = cv2.resize(img_body_peace, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_ok = cv2.resize(img_body_ok, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_wave = cv2.resize(img_body_wave, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_excited = cv2.resize(img_body_excited, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_thumbsup = cv2.resize(img_body_thumbsup, (0,0), fx=scale_factor_body, fy=scale_factor_body)

    ### --- BARU: Muat gambar background virtual --- ###
    # Ganti 'background_kamar.jpg' dengan file gambarmu
    img_virtual_bg = cv2.imread('background_kamar.jpg') 
    if img_virtual_bg is None:
        print("Warning: File background virtual 'background_kamar.jpg' tidak ditemukan. Mode BG Gambar tidak akan bekerja.")

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Error saat memuat gambar: {e}")
    exit()


# --- PENGATURAN POSISI GLOBAL ---
GLOBAL_OFFSET_HEAD = (0, 140)
GLOBAL_OFFSET_BODY = (0, -20)

# --- KONFIGURASI UNTUK SETIAP GESTUR ---
GESTURE_CONFIG = {
    # ... (Konfigurasi gestur tidak berubah)
    "NORMAL": {"head": img_head_normal, "body": img_body_normal},
    "BLUSHING": {"head": img_head_blush, "body": img_body_coverface},
    "EXCITED": {"head": img_head_excited, "body": img_body_excited},
    "THUMBS_UP": {"head": img_head_normal, "body": img_body_thumbsup},
    "PEACE": {"head": img_head_normal, "body": img_body_peace},
    "OK": {"head": img_head_normal, "body": img_body_ok},
    "WAVE": {"head": img_head_normal, "body": img_body_wave}
}

# --- FUNGSI BANTUAN DETEKSI GESTURE ---
def get_finger_status(hand_landmarks):
    # ... (Fungsi ini tidak berubah)
    status = {'THUMB': False, 'INDEX': False, 'MIDDLE': False, 'RING': False, 'PINKY': False}
    if not hand_landmarks:
        return status
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP] 
    if thumb_tip.y < thumb_ip.y:
        status['THUMB'] = True
    finger_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    finger_pips_ids = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
    for i, (tip_id, pip_id) in enumerate(zip(finger_tips_ids, finger_pips_ids)):
        finger_name = list(status.keys())[i+1]
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y:
            status[finger_name] = True
    return status

# --- BUKA KAMERA & INISIALISASI ---
cap = cv2.VideoCapture(0)
history_head = deque(maxlen=5)
history_body = deque(maxlen=5)
gesture_buffer = deque(maxlen=10)
stable_gesture = "NORMAL"

### --- DIUBAH: INISIALISASI MODE BACKGROUND --- ###
# 0 = GREEN SCREEN, 1 = VIRTUAL BACKGROUND (Gambar)
background_mode = 0 
modes = ['BG: GREEN SCREEN (untuk OBS)', 'BG: VIRTUAL IMAGE']
# Variabel untuk menyimpan background yang sudah di-resize
resized_virtual_bg = None
### ------------------------------------------- ###

### --- DIUBAH: Aktifkan 'enable_segmentation' (walau tidak dipakai) --- ###
# Kita biarkan True, tapi kita tidak akan menggunakannya untuk 'np.where' lagi
with mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    enable_segmentation=True) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read() # 'frame' HANYA untuk input
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Proses 'frame' untuk dapatkan 'results', tapi 'frame' tidak akan digambar
        results = holistic.process(image_rgb)
        
        
        ### --- DIUBAH: LOGIKA VIRTUAL BACKGROUND --- ###
        
        # 1. Buat KANVAS BARU (final_output)
        if background_mode == 0: # MODE GREEN SCREEN
            final_output = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            final_output[:] = (0, 255, 0) # BGR untuk Hijau
        
        elif background_mode == 1: # MODE VIRTUAL BACKGROUND (Gambar)
            if img_virtual_bg is not None:
                # Cek apakah background sudah di-resize, jika belum, resize
                if resized_virtual_bg is None or resized_virtual_bg.shape[0] != frame_h or resized_virtual_bg.shape[1] != frame_w:
                    resized_virtual_bg = cv2.resize(img_virtual_bg, (frame_w, frame_h))
                final_output = resized_virtual_bg.copy()
            else:
                # Fallback jika gambar gagal dimuat
                final_output = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                final_output[:] = (20, 20, 20) # Background abu-abu gelap
        
        ### ---------------------------------------- ###
        

        # --- LOGIKA DETEKSI GESTURE ---
        # ... (Logika ini tidak berubah, tetap pakai 'results' dari 'frame' asli)
        current_gesture = "NORMAL"
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            if results.left_hand_landmarks and results.right_hand_landmarks:
                left_wrist = pose[mp_holistic.PoseLandmark.LEFT_WRIST]
                right_wrist = pose[mp_holistic.PoseLandmark.RIGHT_WRIST]
                nose = pose[mp_holistic.PoseLandmark.NOSE]
                if left_wrist.y < nose.y and right_wrist.y < nose.y:
                    if abs(left_wrist.x - right_wrist.x) < 0.3:
                        current_gesture = "BLUSHING"
            if current_gesture == "NORMAL":
                is_right_wave = (results.right_hand_landmarks and pose[mp_holistic.PoseLandmark.RIGHT_WRIST].y < pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y)
                is_left_wave = (results.left_hand_landmarks and pose[mp_holistic.PoseLandmark.LEFT_WRIST].y < pose[mp_holistic.PoseLandmark.LEFT_SHOULDER].y)
                if is_right_wave and is_left_wave:
                    current_gesture = "EXCITED"
                elif is_right_wave or is_left_wave:
                    current_gesture = "WAVE"
                elif results.right_hand_landmarks:
                    hand_landmarks = results.right_hand_landmarks
                    finger_status = get_finger_status(hand_landmarks)
                    if finger_status['THUMB'] and not finger_status['INDEX'] and not finger_status['MIDDLE']:
                        current_gesture = "THUMBS_UP"
                    elif finger_status['INDEX'] and finger_status['MIDDLE'] and not finger_status['RING']:
                        current_gesture = "PEACE"
                    elif math.hypot(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y) < 0.07:
                        current_gesture = "OK"

        # --- LOGIKA STABILITAS GESTURE (DEBOUNCE) ---
        gesture_buffer.append(current_gesture)
        if gesture_buffer.count(current_gesture) > 7:
            stable_gesture = current_gesture
        
        # --- AMBIL KONFIGURASI & GAMBAR KARAKTER ---
        config = GESTURE_CONFIG.get(stable_gesture, GESTURE_CONFIG["NORMAL"])
        current_head_img = config["head"]
        current_body_img = config["body"]
        
        offset_head_x, offset_head_y = GLOBAL_OFFSET_HEAD
        offset_body_x, offset_body_y = GLOBAL_OFFSET_BODY

        # --- LOGIKA PENGGAMBARAN KARAKTER ---
        # Hitung posisi Margo BUKAN dari 'final_output', tapi dari 'results'
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            nose = pose[mp_holistic.PoseLandmark.NOSE]
            left_shoulder = pose[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            raw_pos_head = (int(nose.x * frame_w), int(nose.y * frame_h))
            raw_pos_body = (int((left_shoulder.x + right_shoulder.x) * frame_w / 2), int((left_shoulder.y + right_shoulder.y) * frame_h / 2))
            history_head.append(raw_pos_head)
            history_body.append(raw_pos_body)
            avg_x_head = sum(p[0] for p in history_head) // len(history_head)
            avg_y_head = sum(p[1] for p in history_head) // len(history_head)
            avg_x_body = sum(p[0] for p in history_body) // len(history_body)
            avg_y_body = sum(p[1] for p in history_body) // len(history_body)
            current_body_h, current_body_w, _ = current_body_img.shape
            pos_x_body = avg_x_body - (current_body_w // 2) + offset_body_x
            pos_y_body = avg_y_body - (current_body_h // 2) + offset_body_y
            current_head_h, current_head_w, _ = current_head_img.shape
            pos_x_head = avg_x_head - (current_head_w // 2) + offset_head_x
            pos_y_head = avg_y_head - (current_head_h // 2) + offset_head_y

            ### --- DIUBAH: Gambar Margo di atas 'final_output' --- ###
            if stable_gesture == "BLUSHING":
                final_output = overlay_png(final_output, current_head_img, pos_x_head, pos_y_head)
                final_output = overlay_png(final_output, current_body_img, pos_x_body, pos_y_body)
            else:
                final_output = overlay_png(final_output, current_body_img, pos_x_body, pos_y_body)
                final_output = overlay_png(final_output, current_head_img, pos_x_head, pos_y_head)
        
        # --- TAMPILKAN STATUS BACKGROUND ---
        cv2.putText(final_output, modes[background_mode], (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(final_output, "Tekan 'b' ganti BG", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        ### --- DIUBAH: Tampilkan 'final_output' --- ###
        # Jendela ini sekarang TIDAK menampilkan videomu sama sekali
        cv2.imshow('VTuber Margo Interaktif (HANYA AVATAR)', final_output)
        
        
        # --- LOGIKA TOMBOL ---
        key = cv2.waitKey(5) & 0xFF
        
        if key in [ord('q'), 27]: # Tombol Q atau ESC untuk keluar
            break
        elif key == ord('b'): # Tombol 'b' untuk ganti background
            background_mode = (background_mode + 1) % 2 # Siklus 0 -> 1 -> 0
            
cap.release()
cv2.destroyAllWindows()