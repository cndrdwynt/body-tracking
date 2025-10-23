import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import pyaudio  # <-- Pastikan sudah pip install PyAudio
import audioop  # <-- Ini bawaan Python

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

# --- INISIALISASI MEDIAPIPE ---
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

# --- MUAT SEMUA GAMBAR KARAKTER DAN SKALAKAN ---
try:
    ### --- DIUBAH: Ganti nama aset dasar --- ###
    img_head_normal_base = cv2.imread('margo_head_normal.png', cv2.IMREAD_UNCHANGED)
    img_head_blush_base = cv2.imread('margo_head_blush.png', cv2.IMREAD_UNCHANGED)
    img_head_excited_base = cv2.imread('margo_head_excited.png', cv2.IMREAD_UNCHANGED) 
    
    ### --- BARU: Muat aset ekspresi wajah (sesuai yang kamu punya) --- ###
    img_head_normal_blink = cv2.imread('margo_head_normal_blink.png', cv2.IMREAD_UNCHANGED)
    img_head_normal_open = cv2.imread('margo_head_normal_open.png', cv2.IMREAD_UNCHANGED)
    img_head_blush_blink = cv2.imread('margo_head_blush_blink.png', cv2.IMREAD_UNCHANGED)

    # Aset Badan (7) - (Tidak berubah)
    img_body_normal = cv2.imread('margo_body_normal.png', cv2.IMREAD_UNCHANGED)
    img_body_coverface = cv2.imread('margo_body_coverface.png', cv2.IMREAD_UNCHANGED)
    img_body_peace = cv2.imread('margo_body_peace.png', cv2.IMREAD_UNCHANGED)
    img_body_ok = cv2.imread('margo_body_ok.png', cv2.IMREAD_UNCHANGED)
    img_body_wave = cv2.imread('margo_body_wave.png', cv2.IMREAD_UNCHANGED)
    img_body_excited = cv2.imread('margo_body_excited.png', cv2.IMREAD_UNCHANGED)
    img_body_thumbsup = cv2.imread('margo_body_thumbsup.png', cv2.IMREAD_UNCHANGED)
    
    ### --- DIUBAH: Tambahkan semua aset ke list pengecekan --- ###
    image_assets = [
        img_head_normal_base, img_head_blush_base, img_head_excited_base,
        img_head_normal_blink, img_head_normal_open, img_head_blush_blink, # <-- Aset baru
        img_body_normal, img_body_coverface, img_body_peace, img_body_ok,
        img_body_wave, img_body_excited, img_body_thumbsup
    ]
    if any(img is None for img in image_assets):
        raise FileNotFoundError("Salah satu atau lebih file gambar tidak ditemukan! Cek kembali 13 file asetmu.")
    
    scale_factor_head = 0.4
    scale_factor_body = 0.5 

    # Skalakan semua aset kepala
    img_head_normal_base = cv2.resize(img_head_normal_base, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_blush_base = cv2.resize(img_head_blush_base, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_excited_base = cv2.resize(img_head_excited_base, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_normal_blink = cv2.resize(img_head_normal_blink, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_normal_open = cv2.resize(img_head_normal_open, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    img_head_blush_blink = cv2.resize(img_head_blush_blink, (0,0), fx=scale_factor_head, fy=scale_factor_head)
    
    # Skalakan semua aset badan
    img_body_normal = cv2.resize(img_body_normal, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_coverface = cv2.resize(img_body_coverface, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_peace = cv2.resize(img_body_peace, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_ok = cv2.resize(img_body_ok, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_wave = cv2.resize(img_body_wave, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_excited = cv2.resize(img_body_excited, (0,0), fx=scale_factor_body, fy=scale_factor_body)
    img_body_thumbsup = cv2.resize(img_body_thumbsup, (0,0), fx=scale_factor_body, fy=scale_factor_body)

    # Muat background virtual (Tidak berubah)
    img_virtual_bg = cv2.imread('background_kamar.jpg') 
    if img_virtual_bg is None:
        print("Warning: File background virtual 'background_kamar.jpg' tidak ditemukan.")

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Error saat memuat gambar: {e}")
    exit()

### --- BARU: PETA ASET KEPALA (disesuaikan dengan asetmu) --- ###
HEAD_ASSET_MAP = {
    id(img_head_normal_base): {
        "blink": img_head_normal_blink,
        "open": img_head_normal_open  # Punya
    },
    id(img_head_blush_base): {
        "blink": img_head_blush_blink, # Punya
        "open": None                   # Tidak Punya -> akan di-fallback ke normal_open
    },
    id(img_head_excited_base): {
        "blink": None,                 # Tidak Punya
        "open": None                   # Tidak Punya (karena base-nya sudah 'open')
    }
}

# --- PENGATURAN POSISI GLOBAL --- (Tidak berubah)
GLOBAL_OFFSET_HEAD = (0, 140)
GLOBAL_OFFSET_BODY = (0, -20)

# --- KONFIGURASI UNTUK SETIAP GESTUR ---
### --- DIUBAH: Gunakan aset '_base' --- ###
GESTURE_CONFIG = {
    "NORMAL": {"head": img_head_normal_base, "body": img_body_normal},
    "BLUSHING": {"head": img_head_blush_base, "body": img_body_coverface},
    "EXCITED": {"head": img_head_excited_base, "body": img_body_excited},
    "THUMBS_UP": {"head": img_head_normal_base, "body": img_body_thumbsup},
    "PEACE": {"head": img_head_normal_base, "body": img_body_peace},
    "OK": {"head": img_head_normal_base, "body": img_body_ok},
    "WAVE": {"head": img_head_normal_base, "body": img_body_wave}
}

### --- BARU: Konstanta dan Fungsi Helper Ekspresi Wajah --- ###
EAR_THRESHOLD = 0.3 # <-- SESUAIKAN ANGKA INI

def get_aspect_ratio(face_landmarks, top_idx, bottom_idx, left_idx, right_idx):
    """Menghitung rasio jarak vertikal terhadap horizontal dari 4 landmark."""
    if not face_landmarks:
        return 0.0
    
    lm = face_landmarks.landmark
    
    # Helper untuk mendapatkan (x, y) dari landmark
    def get_p(idx):
        return (lm[idx].x, lm[idx].y)

    try:
        top_p = get_p(top_idx)
        bottom_p = get_p(bottom_idx)
        left_p = get_p(left_idx)
        right_p = get_p(right_idx)
        
        ver_dist = math.hypot(top_p[0] - bottom_p[0], top_p[1] - bottom_p[1])
        hor_dist = math.hypot(left_p[0] - right_p[0], left_p[1] - right_p[1])
        
        if hor_dist < 1e-6: # Menghindari pembagian dengan nol
            return 0.0
        return ver_dist / hor_dist
        
    except IndexError:
        return 0.0
### -------------------------------------------------------- ###


# --- FUNGSI BANTUAN DETEKSI GESTURE --- (Tidak berubah)
def get_finger_status(hand_landmarks):
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

# --- BUKA KAMERA & INISIALISASI --- (Tidak berubah)
cap = cv2.VideoCapture(0)
history_head = deque(maxlen=5)
history_body = deque(maxlen=5)
gesture_buffer = deque(maxlen=10)
stable_gesture = "NORMAL"
background_mode = 0 
modes = ['BG: GREEN SCREEN (untuk OBS)', 'BG: VIRTUAL IMAGE']
resized_virtual_bg = None

### --- BARU: Inisialisasi Audio Input --- ###
CHUNK = 1024                 # Ukuran buffer audio
FORMAT = pyaudio.paInt16     # Format audio 16-bit
CHANNELS = 1                 # Mono
RATE = 44100                 # Sample rate
AUDIO_THRESHOLD = 100       # <-- SESUAIKAN ANGKA INI
talk_frame_counter = 0       # Untuk animasi "flap"

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
### -------------------------------------- ###


# --- BUKA HOLISTIC --- (Tidak berubah)
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
        
        results = holistic.process(image_rgb)
        
        
        # --- LOGIKA VIRTUAL BACKGROUND --- (Tidak berubah)
        if background_mode == 0: # MODE GREEN SCREEN
            final_output = np.zeros((frame_h, frame_w, 3), dtype=np.uint8); final_output[:] = (0, 255, 0)
        elif background_mode == 1: # MODE VIRTUAL BACKGROUND (Gambar)
            if img_virtual_bg is not None:
                if resized_virtual_bg is None or resized_virtual_bg.shape[0] != frame_h or resized_virtual_bg.shape[1] != frame_w:
                    resized_virtual_bg = cv2.resize(img_virtual_bg, (frame_w, frame_h))
                final_output = resized_virtual_bg.copy()
            else:
                final_output = np.zeros((frame_h, frame_w, 3), dtype=np.uint8); final_output[:] = (20, 20, 20)
        
        
        ### --- BARU: DETEKSI BERBICARA (AUDIO) + DEBUG --- ###
        is_talking = False
        if stream: # Hanya jika stream berhasil diinisialisasi
            try:
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                rms = audioop.rms(audio_data, 2) # 2 = 16-bit
                
                print(f"Volume Suara: {rms}") # <-- BARIS DEBUG AUDIO
                
                if rms > AUDIO_THRESHOLD:
                    is_talking = True
            except IOError:
                pass # Ini wajar terjadi
        ### ----------------------------------------------- ###
        
        ### --- BARU: DETEKSI MATA BERKEDIP (EAR) + DEBUG --- ###
        is_blinking = False
        if results.face_landmarks:
            face_lm = results.face_landmarks
            left_ear = get_aspect_ratio(face_lm, 386, 374, 362, 263)
            right_ear = get_aspect_ratio(face_lm, 159, 145, 133, 33)
            avg_ear = (left_ear + right_ear) / 2.0
            
            print(f"Rasio Mata: {avg_ear}") # <-- BARIS DEBUG MATA
            
            if avg_ear < EAR_THRESHOLD:
                is_blinking = True
        ### ---------------------------------------------- ###
        
        # --- LOGIKA DETEKSI GESTURE --- (Tidak berubah)
        current_gesture = "NORMAL"
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            if results.left_hand_landmarks and results.right_hand_landmarks:
                left_wrist = pose[mp_holistic.PoseLandmark.LEFT_WRIST]; right_wrist = pose[mp_holistic.PoseLandmark.RIGHT_WRIST]
                nose = pose[mp_holistic.PoseLandmark.NOSE]
                if left_wrist.y < nose.y and right_wrist.y < nose.y:
                    if abs(left_wrist.x - right_wrist.x) < 0.3: current_gesture = "BLUSHING"
            if current_gesture == "NORMAL":
                is_right_wave = (results.right_hand_landmarks and pose[mp_holistic.PoseLandmark.RIGHT_WRIST].y < pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y)
                is_left_wave = (results.left_hand_landmarks and pose[mp_holistic.PoseLandmark.LEFT_WRIST].y < pose[mp_holistic.PoseLandmark.LEFT_SHOULDER].y)
                if is_right_wave and is_left_wave: current_gesture = "EXCITED"
                elif is_right_wave or is_left_wave: current_gesture = "WAVE"
                elif results.right_hand_landmarks:
                    hand_landmarks = results.right_hand_landmarks
                    finger_status = get_finger_status(hand_landmarks)
                    if finger_status['THUMB'] and not finger_status['INDEX'] and not finger_status['MIDDLE']: current_gesture = "THUMBS_UP"
                    elif finger_status['INDEX'] and finger_status['MIDDLE'] and not finger_status['RING']: current_gesture = "PEACE"
                    elif math.hypot(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y) < 0.07: current_gesture = "OK"

        # --- LOGIKA STABILITAS GESTURE (DEBOUNCE) --- (Tidak berubah)
        gesture_buffer.append(current_gesture)
        if gesture_buffer.count(current_gesture) > 7:
            stable_gesture = current_gesture
        
        
        ### --- DIUBAH: Logika Memilih Aset Kepala (Sesuai Asetmu) --- ###
        config = GESTURE_CONFIG.get(stable_gesture, GESTURE_CONFIG["NORMAL"])
        current_body_img = config["body"]
        base_head_img = config["head"] # Ini adalah gambar kepala _base
        
        current_head_img = base_head_img # Default
        
        variations = HEAD_ASSET_MAP.get(id(base_head_img))

        # Prioritas 1: Gestur 'EXCITED' ("Woo!")
        if stable_gesture == "EXCITED":
            current_head_img = img_head_excited_base # Aset ini sudah 'open', abaikan audio/blink
        
        # Prioritas 2: Berkedip (EAR)
        elif is_blinking and variations and variations.get("blink") is not None:
            # Jika 'blink' ada (normal_blink, blush_blink), gunakan itu
            current_head_img = variations["blink"]
        
        # Prioritas 3: Berbicara (Audio)
        elif is_talking:
            talk_frame_counter = (talk_frame_counter + 1) % 6 # Siklus 6 frame (3 open, 3 base)
            
            # Kita pakai 'img_head_normal_open' sebagai aset "mulut terbuka" universal
            if talk_frame_counter < 3:
                # 3 frame pertama: Mulut terbuka
                current_head_img = img_head_normal_open
            else:
                # 3 frame berikutnya: Mulut tertutup (kembali ke 'base' apa pun itu)
                current_head_img = base_head_img
        
        # Jika tidak ada kondisi di atas, 'current_head_img' tetap 'base_head_img'
        ### -------------------------------------------------- ###
        
        
        offset_head_x, offset_head_y = GLOBAL_OFFSET_HEAD
        offset_body_x, offset_body_y = GLOBAL_OFFSET_BODY

        # --- LOGIKA PENGGAMBARAN KARAKTER --- (Tidak berubah)
        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            nose = pose[mp_holistic.PoseLandmark.NOSE]
            left_shoulder = pose[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            raw_pos_head = (int(nose.x * frame_w), int(nose.y * frame_h))
            raw_pos_body = (int((left_shoulder.x + right_shoulder.x) * frame_w / 2), int((left_shoulder.y + right_shoulder.y) * frame_h / 2))
            history_head.append(raw_pos_head); history_body.append(raw_pos_body)
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

            # Gambar Margo di atas 'final_output'
            if stable_gesture == "BLUSHING":
                final_output = overlay_png(final_output, current_head_img, pos_x_head, pos_y_head)
                final_output = overlay_png(final_output, current_body_img, pos_x_body, pos_y_body)
            else:
                final_output = overlay_png(final_output, current_body_img, pos_x_body, pos_y_body)
                final_output = overlay_png(final_output, current_head_img, pos_x_head, pos_y_head)
        
        # --- TAMPILKAN STATUS BACKGROUND --- (Tidak berubah)
        cv2.putText(final_output, modes[background_mode], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(final_output, "Tekan 'b' ganti BG", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('VTuber Margo Interaktif (HANYA AVATAR)', final_output)
        
        # --- LOGIKA TOMBOL --- (Tidak berubah)
        key = cv2.waitKey(5) & 0xFF
        if key in [ord('q'), 27]: break
        elif key == ord('b'): background_mode = (background_mode + 1) % 2
            
### --- BARU: Matikan audio stream --- ###
if stream:
    stream.stop_stream()
    stream.close()
    p.terminate()
### ---------------------------------- ###

cap.release()
cv2.destroyAllWindows()