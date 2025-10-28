import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import pyaudio 
import audioop 
import random

# --- FUNGSI UNTUK MENEMPELKAN GAMBAR PNG (DENGAN TRANSPARANSI) ---
def overlay_png(background, overlay, x, y):
    # ... (Fungsi ini tidak berubah)
    bg_h, bg_w, _ = background.shape
    overlay_h, overlay_w, _ = overlay.shape
    if x >= bg_w or y >= bg_h or x + overlay_w <= 0 or y + overlay_h <= 0: return background
    y1, y2 = max(0, y), min(bg_h, y + overlay_h)
    x1, x2 = max(0, x), min(bg_w, x + overlay_w)
    alpha_h_start = max(0, -y); alpha_h_end = y2 - y1 + alpha_h_start
    alpha_w_start = max(0, -x); alpha_w_end = x2 - x1 + alpha_w_start
    roi = background[y1:y2, x1:x2]
    overlay_area = overlay[alpha_h_start:alpha_h_end, alpha_w_start:alpha_w_end]
    if roi.shape[0] != overlay_area.shape[0] or roi.shape[1] != overlay_area.shape[1]: return background
    alpha = overlay_area[:, :, 3] / 255.0; alpha_inv = 1.0 - alpha
    for c in range(0, 3): roi[:, :, c] = (alpha * overlay_area[:, :, c] + alpha_inv * roi[:, :, c])
    background[y1:y2, x1:x2] = roi
    return background

# --- FUNGSI BARU UNTUK TEKS DENGAN BACKGROUND ---
def draw_fancy_text(img, text, pos, font, scale, color, thickness, bg_color, alpha=0.5, padding=5):
    # ... (Fungsi ini tidak berubah)
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    top_left = (x, y)
    bottom_right = (x + w + padding*2, y + h + baseline + padding*2)
    text_pos = (x + padding, y + h + padding) 
    y1, y2 = max(0, top_left[1]), min(img.shape[0], bottom_right[1])
    x1, x2 = max(0, top_left[0]), min(img.shape[1], bottom_right[0])
    if y1 >= y2 or x1 >= x2: return img
    sub_img = img[y1:y2, x1:x2]
    bg_rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1 - alpha, bg_rect, alpha, 0)
    img[y1:y2, x1:x2] = res
    cv2.putText(img, text, text_pos, font, scale, color, thickness, cv2.LINE_AA)
    return img


# --- INISIALISASI MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

# --- MUAT SEMUA GAMBAR KARAKTER DAN SKALAKAN ---
try:
    # --- Aset Kepala (Base) ---
    img_head_normal_base = cv2.imread('margo_head_normal.png', cv2.IMREAD_UNCHANGED)
    img_head_blush_base = cv2.imread('margo_head_blush.png', cv2.IMREAD_UNCHANGED)
    img_head_excited_base = cv2.imread('margo_head_excited.png', cv2.IMREAD_UNCHANGED) 
    
    # --- Aset Kepala (Variasi Blink/Open) ---
    img_head_normal_blink = cv2.imread('margo_head_normal_blink.png', cv2.IMREAD_UNCHANGED)
    img_head_normal_open = cv2.imread('margo_head_normal_open.png', cv2.IMREAD_UNCHANGED)
    img_head_blush_blink = cv2.imread('margo_head_blush_blink.png', cv2.IMREAD_UNCHANGED)

    # <<< Muat Aset Kepala untuk Tolah-Toleh/Menunduk >>>
    img_head_look_left = cv2.imread('margo_head_look_left.png', cv2.IMREAD_UNCHANGED)
    img_head_look_right = cv2.imread('margo_head_look_right.png', cv2.IMREAD_UNCHANGED)
    img_head_look_down = cv2.imread('margo_head_look_down.png', cv2.IMREAD_UNCHANGED)
    # <<< ---------------------------------------------------- >>>

    # <<< Muat Aset Laugh >>>
    img_head_laugh_base = cv2.imread('margo_head_laugh.png', cv2.IMREAD_UNCHANGED)
    img_body_laugh = cv2.imread('margo_body_laugh.png', cv2.IMREAD_UNCHANGED)
    # <<< ------------------ >>>

    # --- Aset Badan ---
    img_body_normal = cv2.imread('margo_body_normal.png', cv2.IMREAD_UNCHANGED)
    img_body_coverface = cv2.imread('margo_body_coverface.png', cv2.IMREAD_UNCHANGED)
    img_body_peace = cv2.imread('margo_body_peace.png', cv2.IMREAD_UNCHANGED)
    img_body_ok = cv2.imread('margo_body_ok.png', cv2.IMREAD_UNCHANGED)
    img_body_wave = cv2.imread('margo_body_wave.png', cv2.IMREAD_UNCHANGED)
    img_body_excited = cv2.imread('margo_body_excited.png', cv2.IMREAD_UNCHANGED)
    img_body_thumbsup = cv2.imread('margo_body_thumbsup.png', cv2.IMREAD_UNCHANGED)
    
    # --- Daftar Aset untuk Pengecekan ---
    image_assets = [
        img_head_normal_base, img_head_blush_base, img_head_excited_base,
        img_head_normal_blink, img_head_normal_open, img_head_blush_blink, 
        img_head_look_left, img_head_look_right, img_head_look_down,
        img_body_normal, img_body_coverface, img_body_peace, img_body_ok,
        img_body_wave, img_body_excited, img_body_thumbsup,
        img_head_laugh_base, img_body_laugh
    ]
    if any(img is None for img in image_assets):
        raise FileNotFoundError("Salah satu atau lebih file gambar tidak ditemukan! Pastikan ada: margo_head_normal.png, margo_head_blush.png, margo_head_excited.png, margo_head_normal_blink.png, margo_head_normal_open.png, margo_head_blush_blink.png, margo_head_laugh.png, margo_body_laugh.png, margo_body_normal.png, margo_body_coverface.png, margo_body_peace.png, margo_body_ok.png, margo_body_wave.png, margo_body_excited.png, margo_body_thumbsup.png, DAN TIGA GAMBAR BARU INI: margo_head_look_left.png, margo_head_look_right.png, margo_head_look_down.png")
    
    # --- Faktor Skala ---
    scale_factor_head = 0.4
    scale_factor_body = 0.5 

    # --- Skalakan Aset Kepala ---
    img_head_normal_base = cv2.resize(img_head_normal_base, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_blush_base = cv2.resize(img_head_blush_base, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_excited_base = cv2.resize(img_head_excited_base, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_normal_blink = cv2.resize(img_head_normal_blink, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_normal_open = cv2.resize(img_head_normal_open, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_blush_blink = cv2.resize(img_head_blush_blink, (0,0), fx=scale_factor_head, fy=scale_factor_head)

    # <<< Skalakan Aset Kepala untuk Tolah-Toleh/Menunduk >>>
    img_head_look_left = cv2.resize(img_head_look_left, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_look_right = cv2.resize(img_head_look_right, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_look_down = cv2.resize(img_head_look_down, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    # <<< ---------------------------------------------------- >>>

    # <<< Skalakan Aset Laugh >>>
    img_head_laugh_base = cv2.resize(img_head_laugh_base, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    # <<< --------------------- >>>
    
    # --- Skalakan Aset Badan ---
    img_body_normal = cv2.resize(img_body_normal, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_coverface = cv2.resize(img_body_coverface, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_peace = cv2.resize(img_body_peace, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_ok = cv2.resize(img_body_ok, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_wave = cv2.resize(img_body_wave, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_excited = cv2.resize(img_body_excited, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_thumbsup = cv2.resize(img_body_thumbsup, (0,0), fx=scale_factor_body, fy=scale_factor_body)

    # <<< Skalakan Aset Badan Laugh >>>
    img_body_laugh = cv2.resize(img_body_laugh, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    # <<< --------------------------- >>>

    # Muat background virtual
    img_virtual_bg = cv2.imread('background_virtual.jpeg') 
    if img_virtual_bg is None:
        print("Warning: File background virtual 'virtual.jpeg' tidak ditemukan.")

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Error saat memuat gambar: {e}")
    exit()

# --- MAP Variasi Aset Kepala ---
HEAD_ASSET_MAP = {
    id(img_head_normal_base): { "blink": img_head_normal_blink, "open": img_head_normal_open },
    id(img_head_blush_base): { "blink": img_head_blush_blink, "open": None },
    id(img_head_excited_base): { "blink": None, "open": None },
    id(img_head_laugh_base): { "blink": None, "open": None },
    id(img_head_look_left): { "blink": None, "open": None },
    id(img_head_look_right): { "blink": None, "open": None },
    id(img_head_look_down): { "blink": None, "open": None }
}

# --- OFFSET ---
GLOBAL_OFFSET_HEAD = (5, 80) 
BODY_X_OFFSET_FROM_HEAD = 0
BODY_Y_OFFSET_FROM_HEAD = -90

# --- Konfigurasi State Gestur ---
GESTURE_CONFIG = {
    "NORMAL": {"head": img_head_normal_base, "body": img_body_normal},
    "BLUSHING": {"head": img_head_blush_base, "body": img_body_coverface},
    "EXCITED": {"head": img_head_excited_base, "body": img_body_excited},
    "THUMBS_UP": {"head": img_head_normal_base, "body": img_body_thumbsup},
    "PEACE": {"head": img_head_normal_base, "body": img_body_peace},
    "OK": {"head": img_head_normal_base, "body": img_body_ok},
    "WAVE": {"head": img_head_normal_base, "body": img_body_wave},
    "LAUGH": {"head": img_head_laugh_base, "body": img_body_laugh},
    "LOOK_LEFT_IDLE": {"head": img_head_look_left, "body": img_body_normal},
    "LOOK_RIGHT_IDLE": {"head": img_head_look_right, "body": img_body_normal},
    "LOOK_DOWN_IDLE": {"head": img_head_look_down, "body": img_body_normal}
}

# --- Threshold Kedip Mata ---
EAR_THRESHOLD = 0.3 
def get_aspect_ratio(face_landmarks, top_idx, bottom_idx, left_idx, right_idx):
    # ... (Fungsi ini tidak berubah)
    if not face_landmarks: return 0.0
    lm = face_landmarks.landmark
    def get_p(idx): return (lm[idx].x, lm[idx].y)
    try:
        top_p, bottom_p = get_p(top_idx), get_p(bottom_idx)
        left_p, right_p = get_p(left_idx), get_p(right_idx)
        ver_dist = math.hypot(top_p[0] - bottom_p[0], top_p[1] - bottom_p[1])
        hor_dist = math.hypot(left_p[0] - right_p[0], left_p[1] - right_p[1])
        if hor_dist < 1e-6: return 0.0
        return ver_dist / hor_dist
    except IndexError: return 0.0

# --- FUNGSI BANTUAN DETEKSI GESTURE ---
def get_finger_status(hand_landmarks):
    # ... (Fungsi ini tidak berubah)
    status = {'THUMB': False, 'INDEX': False, 'MIDDLE': False, 'RING': False, 'PINKY': False}
    if not hand_landmarks: return status
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP] 
    if thumb_tip.y < thumb_ip.y: status['THUMB'] = True
    finger_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    finger_pips_ids = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
    for i, (tip_id, pip_id) in enumerate(zip(finger_tips_ids, finger_pips_ids)):
        finger_name = list(status.keys())[i+1]
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y:
            status[finger_name] = True
    return status

# --- BUKA KAMERA & INISIALISASI ---
cap = cv2.VideoCapture(0)

# --- Muat Video Background ---
try:
    cap_bg = cv2.VideoCapture('background_video.mp4') 
    if not cap_bg.isOpened():
        print("Warning: File video background 'background_video.mp4' tidak ditemukan atau tidak bisa dibuka.")
        cap_bg = None
except Exception as e:
    print(f"Error saat memuat video background: {e}")
    cap_bg = None

# --- Inisialisasi Gerakan Pegas/Lerp ---
ret, frame_awal = cap.read()
if not ret:
    print("Error: Tidak bisa membaca frame dari kamera.")
    exit()
frame_h, frame_w, _ = frame_awal.shape
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
current_anchor_pos = (frame_w // 2, frame_h // 2) 
SPRING_FACTOR_POS = 0.15

# <<< MODIFIKASI: Inisialisasi Idle State Machine (SEKARANG SEKUENSIAL) >>>
# Timer awal (5-7 detik sebelum mulai sekuens)
INITIAL_IDLE_DELAY = random.randint(60, 120) # Kurangi dari 150-210
idle_timer = INITIAL_IDLE_DELAY 
idle_sequence_index = -1 

# Urutan state yang diinginkan (TETAP SAMA)
IDLE_SEQUENCE = [
    "NORMAL", 
    "LOOK_LEFT_IDLE", 
    "NORMAL", 
    "LOOK_RIGHT_IDLE", 
    "NORMAL", 
    "LOOK_DOWN_IDLE", 
    "NORMAL" 
]
# Durasi untuk setiap state dalam sekuens (PERPENDEK SEMUA)
IDLE_STATE_DURATION = {
    # Kembali ke NORMAL hanya sebentar (0.5 - 1 detik)
    "NORMAL": random.randint(15, 30), 
    # Menoleh/Menunduk lebih singkat (1.5 - 3 detik)
    "LOOK_LEFT_IDLE": random.randint(20, 45), 
    "LOOK_RIGHT_IDLE": random.randint(20, 45), 
    "LOOK_DOWN_IDLE": random.randint(20, 45)  
}
# Faktor pegas untuk sway sisa saat idle (TETAP SAMA atau bisa diubah nanti)
SPRING_FACTOR_IDLE_SWAY = 0.02 
current_idle_offset_for_sway = (0,0)
# <<< ---------------------------------------------------------------------- >>>


# --- Inisialisasi Variabel Lain ---
gesture_buffer = deque(maxlen=10)
frame_counter = 0 
stable_gesture = "NORMAL" # Mulai dari NORMAL
user_is_idle = True # Flag untuk menandakan user sedang diam

# Variabel untuk deteksi tawa
laugh_audio_counter = 0

background_mode = 0 
modes = ['BG: VIRTUAL IMAGE', 'BG: VIRTUAL VIDEO', 'BG: GREEN SCREEN']
resized_virtual_bg = None

# Inisialisasi Kedip Acak
random_blink_timer = random.randint(60, 150)
RANDOM_BLINK_DURATION = 6
random_blink_counter = 0

# Inisialisasi Audio (Sesuai settinganmu)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
AUDIO_THRESHOLD = 100 
talk_frame_counter = 0 

try:
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    start=False) 
    stream.start_stream()
    print("Mikrofon ditemukan dan dimulai.")
except Exception as e:
    print(f"Error saat memulai PyAudio: {e}")
    print("Pastikan mikrofon terpasang. Fitur berbicara tidak akan berfungsi.")
    stream = None

# --- LOOP UTAMA ---
with mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    enable_segmentation=True) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_counter += 1

        # Reset flag user_is_idle, akan diset True lagi nanti jika tidak ada input
        user_is_idle = True 

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape 
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(image_rgb)
        
        # --- (Blok Background - Tidak Berubah) ---
        # ... (kode backgroundmu) ...
        if background_mode == 0:
            if img_virtual_bg is not None:
                if resized_virtual_bg is None or resized_virtual_bg.shape[0] != frame_h or resized_virtual_bg.shape[1] != frame_w:
                    resized_virtual_bg = cv2.resize(img_virtual_bg, (frame_w, frame_h))
                final_output = resized_virtual_bg.copy()
            else:
                final_output = np.zeros((frame_h, frame_w, 3), dtype=np.uint8); final_output[:] = (20, 20, 20)
        elif background_mode == 1:
             if cap_bg is not None:
                ret_bg, frame_bg = cap_bg.read()
                if not ret_bg:
                    cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_bg, frame_bg = cap_bg.read()
                if ret_bg:
                    final_output = cv2.resize(frame_bg, (frame_w, frame_h))
                else:
                    final_output = np.zeros((frame_h, frame_w, 3), dtype=np.uint8); final_output[:] = (20, 20, 20)
             else:
                final_output = np.zeros((frame_h, frame_w, 3), dtype=np.uint8); final_output[:] = (20, 20, 20)
        elif background_mode == 2:
             final_output = np.zeros((frame_h, frame_w, 3), dtype=np.uint8); final_output[:] = (0, 255, 0)
        
        # --- (Blok Audio Deteksi Tawa) ---
        is_talking = False
        is_laugh_detected = False 
        
        if stream: 
            try:
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                rms = audioop.rms(audio_data, 2)
                
                if rms > AUDIO_THRESHOLD:
                    is_talking = True
                    user_is_idle = False # Ada suara = tidak idle
                    
                if rms > (AUDIO_THRESHOLD * 1.8):
                    laugh_audio_counter = min(laugh_audio_counter + 1, 20)
                else:
                    laugh_audio_counter = max(laugh_audio_counter - 2, 0)
                    
                if laugh_audio_counter > 8:
                    is_laugh_detected = True
                    is_talking = False 
                    user_is_idle = False # Tertawa = tidak idle
                    
            except IOError: pass
        # --- ------------------------ ---

        # --- (Blok Blink) ---
        is_blinking = False
        if results.face_landmarks:
            face_lm = results.face_landmarks
            left_ear = get_aspect_ratio(face_lm, 386, 374, 362, 263)
            right_ear = get_aspect_ratio(face_lm, 159, 145, 133, 33)
            avg_ear = (left_ear + right_ear) / 2.0
            if avg_ear < EAR_THRESHOLD:
                is_blinking = True
                # user_is_idle = False # Opsional: anggap kedip = tidak idle?
        # --- ------------- ---
        
        # --- (Blok Gesture) ---
        user_gesture_detected = "NORMAL" 
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            # Cek apakah hidung terdeteksi (tanda user ada di depan kamera)
            nose_landmark = pose[mp_holistic.PoseLandmark.NOSE]
            if nose_landmark.visibility < 0.5: # Jika hidung tidak terlihat jelas
                user_is_idle = False # Anggap tidak idle jika user tidak jelas
            
            # Deteksi gesture tangan
            if results.left_hand_landmarks and results.right_hand_landmarks:
                 left_wrist = pose[mp_holistic.PoseLandmark.LEFT_WRIST]; right_wrist = pose[mp_holistic.PoseLandmark.RIGHT_WRIST]
                 nose = pose[mp_holistic.PoseLandmark.NOSE]
                 if left_wrist.y < nose.y and right_wrist.y < nose.y:
                    if abs(left_wrist.x - right_wrist.x) < 0.3: user_gesture_detected = "BLUSHING"
            if user_gesture_detected == "NORMAL": 
                is_right_wave = (results.right_hand_landmarks and pose[mp_holistic.PoseLandmark.RIGHT_WRIST].y < pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y)
                is_left_wave = (results.left_hand_landmarks and pose[mp_holistic.PoseLandmark.LEFT_WRIST].y < pose[mp_holistic.PoseLandmark.LEFT_SHOULDER].y)
                if is_right_wave and is_left_wave: user_gesture_detected = "EXCITED"
                elif is_right_wave or is_left_wave: user_gesture_detected = "WAVE"
                elif results.right_hand_landmarks:
                    hand_landmarks = results.right_hand_landmarks
                    finger_status = get_finger_status(hand_landmarks)
                    if finger_status['THUMB'] and not finger_status['INDEX'] and not finger_status['MIDDLE']: user_gesture_detected = "THUMBS_UP"
                    elif finger_status['INDEX'] and finger_status['MIDDLE'] and not finger_status['RING']: user_gesture_detected = "PEACE"
                    elif math.hypot(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y) < 0.07: user_gesture_detected = "OK"

        # Stabilisasi Gestur & Cek Idle
        gesture_buffer.append(user_gesture_detected)
        final_user_gesture = "NORMAL" # Gesture user yang stabil
        if gesture_buffer.count(user_gesture_detected) > 7 and user_gesture_detected != "NORMAL":
            final_user_gesture = user_gesture_detected
            user_is_idle = False # Ada gesture = tidak idle
        elif gesture_buffer.count("NORMAL") > 7:
             pass # Biarkan user_is_idle tetap True (jika tidak ada input lain)
        else: # Jika gesture belum stabil
             user_is_idle = False # Anggap tidak idle
        # --- --------------- ---


        # <<< MODIFIKASI UTAMA: Logika Idle State Machine (SEKARANG SEKUENSIAL) >>>
        if user_is_idle:
            idle_timer -= 1
            if idle_timer <= 0:
                # 1. Maju ke state berikutnya dalam sekuens
                idle_sequence_index = (idle_sequence_index + 1) % len(IDLE_SEQUENCE)
                
                # 2. Dapatkan nama state berikutnya
                next_idle_state = IDLE_SEQUENCE[idle_sequence_index] # Misal "LOOK_LEFT_IDLE"
                
                # 3. Set stable_gesture ke state idle ini
                stable_gesture = next_idle_state 

                # 4. Reset timer sesuai durasi state ini
                # Ambil durasi dari dictionary, fallback ke 90 frame jika tidak ada
                duration = IDLE_STATE_DURATION.get(next_idle_state, 90) 
                idle_timer = duration 
        else:
            # Jika user TIDAK idle (bicara, tertawa, gesture), reset sekuens idle
            idle_timer = INITIAL_IDLE_DELAY # Reset timer awal
            idle_sequence_index = -1 # Kembali ke awal sekuens
            # Jika gesture user stabil dan bukan NORMAL, gunakan itu. Jika tidak, default ke NORMAL.
            if final_user_gesture != "NORMAL":
                 stable_gesture = final_user_gesture
            else:
                 stable_gesture = "NORMAL" # Kembali ke normal jika tidak ada gesture kuat
        
        # --- Logika Sway Idle (Goyangan Kecil) ---
        # Simulasikan target offset berdasarkan state idle saat ini
        simulated_target_idle_offset = (0,0)
        current_idle_state_from_sequence = IDLE_SEQUENCE[idle_sequence_index] if idle_sequence_index != -1 else "NORMAL"
        
        if current_idle_state_from_sequence == "LOOK_LEFT_IDLE":
             simulated_target_idle_offset = (-10, 0) 
        elif current_idle_state_from_sequence == "LOOK_RIGHT_IDLE":
             simulated_target_idle_offset = (10, 0)
        elif current_idle_state_from_sequence == "LOOK_DOWN_IDLE":
             simulated_target_idle_offset = (0, 10)
        
        # Hitung lerp untuk sway (hanya jika benar-benar idle)
        if user_is_idle:
            new_idle_x_sway = int(current_idle_offset_for_sway[0] * (1.0 - SPRING_FACTOR_IDLE_SWAY) + simulated_target_idle_offset[0] * SPRING_FACTOR_IDLE_SWAY)
            new_idle_y_sway = int(current_idle_offset_for_sway[1] * (1.0 - SPRING_FACTOR_IDLE_SWAY) + simulated_target_idle_offset[1] * SPRING_FACTOR_IDLE_SWAY)
            current_idle_offset_for_sway = (new_idle_x_sway, new_idle_y_sway)
        else:
            current_idle_offset_for_sway = (0,0) # Reset sway jika tidak idle
        # <<< ----------------------------------------------------------------- >>>


        # --- Blok Pemilihan Aset (Dengan Prioritas) ---
        
        # Prioritas #1: LAUGH 
        if is_laugh_detected:
            stable_gesture = "LAUGH" 
            random_blink_counter = 0; talk_frame_counter = 0
            # Reset idle juga
            idle_timer = INITIAL_IDLE_DELAY; idle_sequence_index = -1
        
        # Ambil konfigurasi (kepala dan badan) berdasarkan stable_gesture final
        config = GESTURE_CONFIG.get(stable_gesture, GESTURE_CONFIG["NORMAL"])
        current_body_img = config["body"]
        base_head_img = config["head"] 
        
        current_head_img = base_head_img # Default
        variations = HEAD_ASSET_MAP.get(id(base_head_img))

        # (Logika Timer Kedip Acak - Tidak Berubah)
        force_blink = False
        # ... (kode kedip acakmu) ...
        if random_blink_counter > 0:
            force_blink = True
            random_blink_counter -= 1
        else:
            random_blink_timer -= 1
            if random_blink_timer <= 0:
                force_blink = True
                random_blink_counter = RANDOM_BLINK_DURATION
                random_blink_timer = random.randint(60, 150)

        # Logika Pemilihan Aset Kepala (Dengan Prioritas)
        if is_laugh_detected:
            current_head_img = img_head_laugh_base
        elif stable_gesture == "EXCITED":
             current_head_img = img_head_excited_base
        elif (is_blinking or force_blink): 
            if variations and variations.get("blink") is not None:
                current_head_img = variations["blink"]
                if is_blinking: random_blink_timer = random.randint(60, 150); random_blink_counter = 0
            else: # Fallback blink
                if stable_gesture == "BLUSHING" and img_head_blush_blink is not None: current_head_img = img_head_blush_blink
                elif img_head_normal_blink is not None: current_head_img = img_head_normal_blink
        elif is_talking: 
            talk_frame_counter = (talk_frame_counter + 1) % 6 
            if talk_frame_counter < 3:
                if variations and variations.get("open") is not None: current_head_img = variations["open"]
                else: # Fallback open
                    if img_head_normal_open is not None: current_head_img = img_head_normal_open
            else: current_head_img = base_head_img
        else: # Default dari config (termasuk state idle LOOK_LEFT, dll.)
            current_head_img = base_head_img 
        # --- ----------------------------------- ---

        
        
        ### --- Logika Penggambaran --- ###
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            nose = pose[mp_holistic.PoseLandmark.NOSE]
            
            # 1. Posisi TARGET (dari hidungmu)
            target_pos_anchor = (int(nose.x * frame_w), int(nose.y * frame_h))
            
            # 2. Lerp Posisi UTAMA
            new_anchor_x = int(current_anchor_pos[0] * (1.0 - SPRING_FACTOR_POS) + target_pos_anchor[0] * SPRING_FACTOR_POS)
            new_anchor_y = int(current_anchor_pos[1] * (1.0 - SPRING_FACTOR_POS) + target_pos_anchor[1] * SPRING_FACTOR_POS)
            current_anchor_pos = (new_anchor_x, new_anchor_y)
            anchor_x, anchor_y = new_anchor_x, new_anchor_y
            
            # 3. Hitung Posisi KEPALA & BADAN
            current_head_h, current_head_w, _ = current_head_img.shape 
            pos_x_head = anchor_x - (current_head_w // 2) + GLOBAL_OFFSET_HEAD[0]
            pos_y_head = anchor_y - (current_head_h // 2) + GLOBAL_OFFSET_HEAD[1]
            current_body_h, current_body_w, _ = current_body_img.shape
            pos_x_body = anchor_x - (current_body_w // 2) + BODY_X_OFFSET_FROM_HEAD
            pos_y_body = pos_y_head + BODY_Y_OFFSET_FROM_HEAD

            # <<< Gerakan "Bernapas" + "Sway Idle" >>>
            breath_sway_y = int(math.sin(frame_counter * 0.05) * 3) 
            breath_sway_x = int(math.cos(frame_counter * 0.20) * 5) 
            total_idle_x_sway = current_idle_offset_for_sway[0] + breath_sway_x
            total_idle_y_sway = current_idle_offset_for_sway[1] + breath_sway_y
            pos_y_head += total_idle_y_sway
            pos_y_body += total_idle_y_sway
            pos_x_head += total_idle_x_sway
            pos_x_body += total_idle_x_sway
            # <<< -------------------------------- >>>
            
            # --- Efek Guncang Saat LAUGH ---
            laugh_shake_x, laugh_shake_y = 0, 0
            if stable_gesture == "LAUGH": 
                laugh_shake_x = int(random.uniform(-15, 15)) 
                laugh_shake_y = int(random.uniform(-10, 10))
            pos_y_head += laugh_shake_y; pos_y_body += laugh_shake_y
            pos_x_head += laugh_shake_x; pos_x_body += laugh_shake_x
            # --- ------------------------ ---
            
            # 5. Gambar Margo
            if stable_gesture == "BLUSHING":
                final_output = overlay_png(final_output, current_head_img, pos_x_head, pos_y_head)
                final_output = overlay_png(final_output, current_body_img, pos_x_body, pos_y_body)
            else:
                final_output = overlay_png(final_output, current_body_img, pos_x_body, pos_y_body)
                final_output = overlay_png(final_output, current_head_img, pos_x_head, pos_y_head)
        ### --------------------------- ###
        
        
        # --- (Blok Teks dan Tampilan - Tidak Berubah) ---
        # ... (kode teksmu) ...
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        text_mode = modes[background_mode]
        pos_mode = (10, 10)
        draw_fancy_text(final_output, text_mode, pos_mode, font_face, 0.7, font_color, 2, bg_color, alpha=0.6)
        text_help = "Tekan 'b' ganti BG | 'q' untuk Keluar"
        pos_help = (10, 50)
        draw_fancy_text(final_output, text_help, pos_help, font_face, 0.5, font_color, 1, bg_color, alpha=0.6)

        # Tampilkan Gambar
        cv2.imshow('VTuber Margo Interaktif (HANYA AVATAR)', final_output)
        
        key = cv2.waitKey(5) & 0xFF
        if key in [ord('q'), 27]: break
        elif key == ord('b'): background_mode = (background_mode + 1) % 3
            
# --- Cleanup ---
# ... (kode cleanupmu) ...
if stream:
    stream.stop_stream()
    stream.close()
    p.terminate()
cap.release()
if cap_bg is not None:
    cap_bg.release()
cv2.destroyAllWindows()