import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import pyaudio
import audioop
import random
import os
import time
import traceback

# --- KONSTANTA BARU: EFFECTS ---
EFFECTS_FOLDER = 'effects'  # Folder untuk efek greenscreen

# Path efek
EFFECT_EXCITED_VIDEO = 'excited_effect.mp4'      # Video saat EXCITED (fullscreen)
EFFECT_LAUGH_VIDEO = 'laugh_effect.mp4'          # Video saat LAUGH (kanan layar)
EFFECT_THUMBSUP_VIDEO = 'thumbsup_effect.mp4'  # <-- BARU: Video saat THUMBS UP (fullscreen)

# Pengaturan Chroma Key (Greenscreen removal)
CHROMA_KEY_COLOR_LOWER = np.array([40, 40, 40])   # HSV lower bound untuk hijau
CHROMA_KEY_COLOR_UPPER = np.array([80, 255, 255]) # HSV upper bound untuk hijau
CHROMA_TOLERANCE = 30  # Toleransi untuk smoothing edge

# --- KONSTANTA ASLI ---
WEBCAM_INDEX = 0
WINDOW_NAME = 'VTuber Margo Interaktif (OpenCV Project)'
EXIT_KEYS = [ord('q'), 27]
BG_CHANGE_KEY = ord('b')

ASSET_FOLDER = '.'
HEAD_SCALE_FACTOR = 0.4
BODY_SCALE_FACTOR = 0.5
HEAD_GLOBAL_OFFSET = (5, 80)
BODY_OFFSET_FROM_HEAD_X = 0
BODY_OFFSET_FROM_HEAD_Y = -90

MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5
FRAME_PROCESS_INTERVAL = 1

POSE_SPRING_FACTOR = 0.15 
HAIR_SPRING_FACTOR = 0.04
HAIR_DRAG_FACTOR = 0.92
HAIR_OFFSET_X = 5
HAIR_OFFSET_Y = 85
HAIR_MAX_DISTANCE = 80
BREATH_SWAY_Y_FREQ = 0.03
BREATH_SWAY_Y_AMP = 2
BREATH_SWAY_X_FREQ = 0.15
BREATH_SWAY_X_AMP = 3
IDLE_SWAY_SPRING_FACTOR = 0.08  
IDLE_SWAY_TARGET_OFFSET = 10
BOUNCE_DECAY = 0.50
BOUNCE_STRENGTH = 8
BOUNCE_GRAVITY = 0.5

# --- KONSTANTA AUDIO & LIP SYNC ---
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHUNK_SIZE = 1024
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16

# Threshold RMS untuk berbagai level suara (DISESUAIKAN LEBIH RENDAH)
AUDIO_SPEAK_THRESHOLD_RMS = 90      # Di bawah ini = diam/mulut tutup
AUDIO_RMS_LEVEL_SMALL = 100         # Suara pelan (mulut sedikit buka)
AUDIO_RMS_LEVEL_MEDIUM = 150        # Suara normal (mulut sedang)
AUDIO_RMS_LEVEL_WIDE = 200          # Suara keras (mulut lebar)
AUDIO_RMS_LEVEL_O = 250             # Untuk 'O' shape (optional)

# Lip sync smoothing
LIP_SYNC_SMOOTH_FACTOR = 0.4        # Seberapa smooth transisi mulut (0.0-1.0)
LIP_SYNC_HOLD_FRAMES = 2            # Berapa frame hold sebelum kembali ke closed

# Deteksi tawa (DIPERMUDAH)
LAUGH_RMS_THRESHOLD = 250           # Threshold absolut untuk tawa (lebih rendah dari sebelumnya)
LAUGH_COUNTER_THRESHOLD = 5         # Cukup 5 frame (0.15 detik) untuk trigger tawa
LAUGH_COUNTER_MAX = 20
LAUGH_COUNTER_DECREMENT = 1         # Decay lebih lambat, agar lebih mudah maintain
LAUGH_SHAKE_AMOUNT_X = 15
LAUGH_SHAKE_AMOUNT_Y = 10
TALK_FRAME_CYCLE = 6  # Legacy, akan diganti sistem baru

EYE_AR_THRESHOLD = 0.3
RANDOM_BLINK_DURATION_FRAMES = 6
RANDOM_BLINK_INTERVAL_MIN_FRAMES = 60
RANDOM_BLINK_INTERVAL_MAX_FRAMES = 150

GESTURE_BUFFER_SIZE = 10
GESTURE_CONFIRMATION_THRESHOLD = 4
GESTURE_OK_DISTANCE_THRESHOLD = 0.15
GESTURE_BLUSH_WRIST_DISTANCE_THRESHOLD = 0.3

INITIAL_IDLE_DELAY_MIN_FRAMES = 30
INITIAL_IDLE_DELAY_MAX_FRAMES = 60
IDLE_NORMAL_DURATION_MIN_FRAMES = 10
IDLE_NORMAL_DURATION_MAX_FRAMES = 20
IDLE_LOOK_DURATION_MIN_FRAMES = 20
IDLE_LOOK_DURATION_MAX_FRAMES = 45
IDLE_SEQUENCE = ["NORMAL", "LOOK_LEFT_IDLE", "NORMAL", "LOOK_RIGHT_IDLE", "NORMAL", "LOOK_DOWN_IDLE", "NORMAL"]
IDLE_STATE_DURATION_RANGES = {
    "NORMAL": (IDLE_NORMAL_DURATION_MIN_FRAMES, IDLE_NORMAL_DURATION_MAX_FRAMES),
    "LOOK_LEFT_IDLE": (IDLE_LOOK_DURATION_MIN_FRAMES, IDLE_LOOK_DURATION_MAX_FRAMES),
    "LOOK_RIGHT_IDLE": (IDLE_LOOK_DURATION_MIN_FRAMES, IDLE_LOOK_DURATION_MAX_FRAMES),
    "LOOK_DOWN_IDLE": (IDLE_LOOK_DURATION_MIN_FRAMES, IDLE_LOOK_DURATION_MAX_FRAMES)
}

UI_FONT = cv2.FONT_HERSHEY_DUPLEX
UI_TEXT_COLOR = (255, 255, 255)
UI_BG_COLOR = (0, 0, 0)
UI_BG_ALPHA = 0.6
UI_PADDING = 5
UI_MODE_POS = (10, 10)
UI_MODE_SCALE = 0.7
UI_MODE_THICKNESS = 2
UI_HELP_POS = (10, 50)
UI_HELP_SCALE = 0.5
UI_HELP_THICKNESS = 1
# RMS Monitor UI
UI_RMS_MONITOR_ENABLED = True       # Set False untuk disable monitor
UI_RMS_POS = (10, 90)               # Posisi RMS monitor
UI_RMS_SCALE = 0.5
UI_RMS_THICKNESS = 1

BG_VIRTUAL_IMAGE_PATH = 'background_virtual.jpeg'
BG_VIRTUAL_VIDEO_PATH = 'background_video.mp4'
BG_MODES = ['BG: VIRTUAL IMAGE', 'BG: VIRTUAL VIDEO', 'BG: GREEN SCREEN']
BG_FALLBACK_COLOR = (20, 20, 20)
BG_GREEN_SCREEN_COLOR = (0, 255, 0)

# --- INISIALISASI MEDIAPIPE ---
try:
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands
except AttributeError:
    print("Error: Gagal init mediapipe.")
    exit()

# --- FUNGSI UTILITAS (Tidak berubah) ---
def overlay_png(background, overlay, x, y):
    """Menempelkan gambar PNG transparan ke background."""
    try:
        if background is None:
            print("Error: Background is None.")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        if overlay is None:
            return background
        bg_h, bg_w, *bg_channels_list = background.shape
        bg_channels = bg_channels_list[0] if bg_channels_list else 1
        overlay_h, overlay_w, *overlay_channels_list = overlay.shape
        overlay_channels = overlay_channels_list[0] if overlay_channels_list else 1
        if bg_channels == 1:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
            bg_h, bg_w, bg_channels = background.shape
        if overlay_channels < 4:
            if overlay_channels == 3:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
                overlay[:, :, 3] = 255
                overlay_h, overlay_w, overlay_channels = overlay.shape
            else:
                return background
        if x >= bg_w or y >= bg_h or x + overlay_w <= 0 or y + overlay_h <= 0:
            return background
        y1, y2 = max(0, y), min(bg_h, y + overlay_h)
        x1, x2 = max(0, x), min(bg_w, x + overlay_w)
        alpha_h_start = max(0, -y)
        alpha_h_end = y2 - y1 + alpha_h_start
        alpha_w_start = max(0, -x)
        alpha_w_end = x2 - x1 + alpha_w_start
        if y1 >= y2 or x1 >= x2:
            return background
        roi = background[y1:y2, x1:x2]
        if alpha_h_start >= alpha_h_end or alpha_w_start >= alpha_w_end or alpha_h_start < 0 or alpha_h_end > overlay_h or alpha_w_start < 0 or alpha_w_end > overlay_w:
            return background
        overlay_area = overlay[alpha_h_start:alpha_h_end, alpha_w_start:alpha_w_end]
        if roi.size == 0 or overlay_area.size == 0:
            return background
        if roi.shape[:2] != overlay_area.shape[:2]:
            try:
                overlay_area = cv2.resize(overlay_area, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
            except Exception as resize_error:
                print(f"Resize Error: {resize_error}")
                return background
        alpha = overlay_area[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(0, 3):
            if roi.shape[2] > c:
                roi[:, :, c] = (alpha * overlay_area[:, :, c] + alpha_inv * roi[:, :, c])
        background[y1:y2, x1:x2] = roi
        return background
    except Exception as e:
        print(f"Overlay Error: {e}")
        return background if background is not None else np.zeros((100, 100, 3), dtype=np.uint8)

# --- FUNGSI BARU: CHROMA KEY (GREENSCREEN REMOVAL) ---
def remove_greenscreen(frame):
    """Hapus background hijau dari video dan kembalikan frame BGRA dengan alpha channel."""
    if frame is None or frame.size == 0:
        return None
    
    try:
        # Convert BGR ke HSV untuk deteksi warna hijau lebih baik
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Buat mask untuk warna hijau
        mask = cv2.inRange(hsv, CHROMA_KEY_COLOR_LOWER, CHROMA_KEY_COLOR_UPPER)
        
        # Inverse mask (1 = keep, 0 = remove)
        mask_inv = cv2.bitwise_not(mask)
        
        # Smooth edges dengan blur
        mask_inv = cv2.GaussianBlur(mask_inv, (5, 5), 0)
        
        # Buat frame BGRA
        b, g, r = cv2.split(frame)
        frame_bgra = cv2.merge([b, g, r, mask_inv])
        
        return frame_bgra
    except Exception as e:
        print(f"Greenscreen removal error: {e}")
        return None

def overlay_video_effect(background, video_frame, position='fullscreen', frame_w=None, frame_h=None):
    """Overlay video effect dengan chroma key ke background.
    
    Args:
        background: Frame background
        video_frame: Frame video effect (akan dihapus greenscreen-nya)
        position: 'fullscreen' atau 'right' (untuk posisi kanan)
        frame_w, frame_h: Ukuran frame output
    """
    if background is None or video_frame is None:
        return background
    
    bg_h, bg_w = background.shape[:2]
    
    # Hapus greenscreen dari video effect
    effect_with_alpha = remove_greenscreen(video_frame)
    if effect_with_alpha is None:
        return background
    
    if position == 'fullscreen':
        # Resize effect ke ukuran penuh layar
        effect_resized = cv2.resize(effect_with_alpha, (bg_w, bg_h))
        return overlay_png(background, effect_resized, 0, 0)
    
    elif position == 'right':
        # Letakkan effect di sebelah kanan (50% lebar layar) - Sesuai kode terakhir
        effect_h, effect_w = effect_with_alpha.shape[:2]
        target_w = int(bg_w * 0.5)  # 50% dari lebar layar
        target_h = int(effect_h * (target_w / effect_w))  # Maintain aspect ratio
        
        # Pastikan tidak melebihi tinggi layar
        if target_h > bg_h:
            target_h = bg_h
            target_w = int(effect_w * (target_h / effect_h))
        
        effect_resized = cv2.resize(effect_with_alpha, (target_w, target_h))
        
        # Posisi: kanan layar, vertikal tengah
        x_pos = bg_w - target_w - 20  # 20px dari kanan
        y_pos = (bg_h - target_h) // 2  # Tengah vertikal
        
        return overlay_png(background, effect_resized, x_pos, y_pos)
    
    return background

def draw_fancy_text(img, text, pos, font, scale, color, thickness, bg_color, alpha, padding):
    """Menggambar teks dengan background."""
    try:
        if img is None:
            return img
        if not isinstance(text, str):
            text = str(text)
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = int(pos[0]), int(pos[1])
        top_left = (x, y)
        bottom_right = (x + w + padding * 2, y + h + baseline + padding * 2)
        text_pos = (x + padding, y + h + padding)
        img_h, img_w, _ = img.shape
        y1, y2 = max(0, top_left[1]), min(img_h, bottom_right[1])
        x1, x2 = max(0, top_left[0]), min(img_w, bottom_right[0])
        if y1 >= y2 or x1 >= x2:
            return img
        sub_img = img[y1:y2, x1:x2]
        if sub_img.size == 0:
            return img
        bg_rect = np.full(sub_img.shape, bg_color, dtype=sub_img.dtype)
        if sub_img.shape != bg_rect.shape:
            return img
        res = cv2.addWeighted(sub_img, 1 - alpha, bg_rect, alpha, 0)
        img[y1:y2, x1:x2] = res
        cv2.putText(img, text, text_pos, font, scale, color, thickness, cv2.LINE_AA)
        return img
    except Exception as e:
        print(f"Draw Text Error ('{text}'): {e}")
        return img

def get_aspect_ratio(face_landmarks, top_idx, bottom_idx, left_idx, right_idx):
    """Hitung rasio aspek."""
    if not face_landmarks:
        return 0.0
    lm = face_landmarks.landmark
    def get_p(idx):
        try:
            return (lm[idx].x, lm[idx].y) if idx < len(lm) and lm[idx].visibility > 0.1 else None
        except IndexError:
            return None
    top_p, bottom_p, left_p, right_p = get_p(top_idx), get_p(bottom_idx), get_p(left_idx), get_p(right_idx)
    if None in [top_p, bottom_p, left_p, right_p]:
        return 0.0
    try:
        ver = math.hypot(top_p[0] - bottom_p[0], top_p[1] - bottom_p[1])
        hor = math.hypot(left_p[0] - right_p[0], left_p[1] - right_p[1])
        return ver / hor if hor > 1e-6 else 0.0
    except Exception as e:
        print(f"Aspect Ratio Error: {e}")
        return 0.0

def get_finger_status(hand_landmarks):
    """Deteksi jari lurus."""
    status = {'THUMB': False, 'INDEX': False, 'MIDDLE': False, 'RING': False, 'PINKY': False}
    
    if not hand_landmarks:
        return status
    
    try:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        if thumb_tip.y < thumb_ip.y:
            status['THUMB'] = True
        
        finger_tips_ids = [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ]
        finger_pips_ids = [
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP
        ]
        
        finger_names = ['INDEX', 'MIDDLE', 'RING', 'PINKY']
        
        for i, (tip_id, pip_id) in enumerate(zip(finger_tips_ids, finger_pips_ids)):
            finger_name = finger_names[i]
            tip = hand_landmarks.landmark[tip_id]
            pip = hand_landmarks.landmark[pip_id]
            
            if tip.y < pip.y:
                status[finger_name] = True
        
    except (IndexError, AttributeError) as e:
        print(f"Warning: Error di get_finger_status: {e}")
    
    return status

# --- FUNGSI INISIALISASI ---
def load_assets(folder):
    """Memuat dan menskalakan semua aset gambar."""
    assets = {}
    asset_files = {
        "head_normal": 'margo_head_normal.png',
        "head_blush": 'margo_head_blush.png',
        "head_excited": 'margo_head_excited.png',
        "head_normal_blink": 'margo_head_normal_blink.png',
        # --- LIP SYNC VARIATIONS ---
        "head_normal_closed": 'margo_head_normal.png',      # Mulut tutup (default)
        "head_normal_small": 'margo_mouth_small.png',       # Mulut sedikit buka
        "head_normal_medium": 'margo_mouth_medium.png',     # Mulut sedang
        "head_normal_wide": 'margo_mouth_wide.png',         # Mulut lebar
        "head_normal_o": 'margo_mouth_o.png',               # Mulut 'O' (optional)
        # ----------------------------
        "head_blush_blink": 'margo_head_blush_blink.png',
        "head_look_left": 'margo_head_look_left.png',
        "head_look_right": 'margo_head_look_right.png',
        "head_look_down": 'margo_head_look_down.png',
        "head_laugh": 'margo_head_laugh.png',
        "hair_back": 'hair_back.png',
        "body_normal": 'margo_body_normal.png',
        "body_coverface": 'margo_body_coverface.png',
        "body_peace": 'margo_body_peace.png',
        "body_ok": 'margo_body_ok.png',
        "body_wave": 'margo_body_wave.png',
        "body_excited": 'margo_body_excited.png',
        "body_thumbsup": 'margo_body_thumbsup.png',
        "body_laugh": 'margo_body_laugh.png',
        "bg_virtual_img": BG_VIRTUAL_IMAGE_PATH
    }
    print("Memuat aset...")
    all_loaded = True
    for name, filename in asset_files.items():
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback untuk mouth variations jika tidak ada
            if "mouth_" in name:
                print(f"Warn: '{filename}' tidak ditemukan, gunakan default")
                assets[name] = assets.get("head_normal")  # Fallback ke normal
                continue
            print(f"Error load: '{filename}'")
            all_loaded = False
            assets[name] = None
            continue
        is_head = "head" in name or "mouth" in name
        is_hair = "hair" in name
        is_body = "body" in name
        is_bg = "bg_" in name
        scale = HEAD_SCALE_FACTOR if (is_head or is_hair) else BODY_SCALE_FACTOR if is_body else 1.0
        if not is_bg:
            try:
                if img.shape[0] > 0 and img.shape[1] > 0:
                    assets[name] = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                else:
                    print(f"Warn: Dimensi 0 '{filename}'")
                    assets[name] = img
            except Exception as e:
                print(f"Error resize '{filename}': {e}")
                assets[name] = None
                all_loaded = False
        else:
            assets[name] = img
        if assets.get(name) is not None:
            print(f" - OK: {filename}")
    if not all_loaded:
        raise FileNotFoundError("Aset penting hilang/gagal.")

    # Membuat Map Variasi Kepala
    assets['head_variations'] = {}
    head_variation_map_config = {
        "head_normal": {
            "blink": "head_normal_blink",
            "closed": "head_normal_closed",
            "small": "head_normal_small",
            "medium": "head_normal_medium",
            "wide": "head_normal_wide",
            "o": "head_normal_o"
        },
        "head_blush": {"blink": "head_blush_blink"},
        "head_excited": {},
        "head_laugh": {},
        "head_look_left": {},
        "head_look_right": {},
        "head_look_down": {}
    }
    for base_name, variation_names in head_variation_map_config.items():
        base_img = assets.get(base_name)
        if base_img is not None:
            variations_for_base = {}
            for var_type, var_name in variation_names.items():
                var_asset = assets.get(var_name)
                if var_asset is not None:
                    variations_for_base[var_type] = var_asset
            assets['head_variations'][id(base_img)] = variations_for_base

    # Membuat Konfigurasi Gestur
    assets['gesture_config'] = {}
    gesture_config_setup = [
        ("NORMAL", "head_normal", "body_normal"),
        ("BLUSHING", "head_blush", "body_coverface"),
        ("EXCITED", "head_excited", "body_excited"),
        ("THUMBS_UP", "head_normal", "body_thumbsup"),
        ("PEACE", "head_normal", "body_peace"),
        ("OK", "head_normal", "body_ok"),
        ("WAVE", "head_normal", "body_wave"),
        ("LAUGH", "head_laugh", "body_laugh"),
        ("LOOK_LEFT_IDLE", "head_look_left", "body_normal"),
        ("LOOK_RIGHT_IDLE", "head_look_right", "body_normal"),
        ("LOOK_DOWN_IDLE", "head_look_down", "body_normal")
    ]
    for gesture_name, head_name, body_name in gesture_config_setup:
        head_asset = assets.get(head_name)
        body_asset = assets.get(body_name)
        if head_asset is not None and body_asset is not None:
            assets['gesture_config'][gesture_name] = {"head": head_asset, "body": body_asset}
        else:
            print(f"Warning: Aset untuk gesture '{gesture_name}' tidak lengkap. Dilewati.")

    if not assets['gesture_config'] or "NORMAL" not in assets['gesture_config']:
        raise ValueError("Konfigurasi gestur NORMAL hilang.")

    print("Aset selesai diproses.")
    return assets

def initialize_systems():
    """Inisialisasi sistem termasuk video effect."""
    cap = stream = audio_system = holistic = cap_bg = None
    cap_effect_excited = cap_effect_laugh = cap_effect_thumbsup = None  # <-- BARU
    frame_h, frame_w = 480, 640

    # Kamera
    try:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            raise IOError(f"Cannot open webcam {WEBCAM_INDEX}")
        ret, frame = cap.read()
        if not ret or frame is None:
            raise IOError("Cannot read initial frame.")
        frame_h, frame_w, _ = frame.shape
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"Kamera {WEBCAM_INDEX} ({frame_w}x{frame_h}) OK.")
    except Exception as e:
        print(f"Error init kamera: {e}")
        if cap:
            cap.release()
        raise

    # Audio
    try:
        audio_system = pyaudio.PyAudio()
        stream = audio_system.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS,
                                   rate=AUDIO_SAMPLE_RATE, input=True,
                                   frames_per_buffer=AUDIO_CHUNK_SIZE, start=True)
        print("Mikrofon OK.")
    except Exception as e:
        print(f"Warn: Gagal init audio: {e}. No audio features.")
        if stream:
            stream.close()
        if audio_system:
            audio_system.terminate()
        audio_system = stream = None

    # MediaPipe
    try:
        holistic = mp_holistic.Holistic(min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
                                        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
                                        enable_segmentation=True)
        print("MediaPipe OK.")
    except Exception as e:
        print(f"Error init MP: {e}")
        if cap:
            cap.release()
        if stream:
            stream.close()
        if audio_system:
            audio_system.terminate()
        raise

    # BG Video
    try:
        path = os.path.join(ASSET_FOLDER, BG_VIRTUAL_VIDEO_PATH)
        if os.path.exists(path):
            cap_bg = cv2.VideoCapture(path)
            if cap_bg is None or not cap_bg.isOpened():
                print(f"Warn: Cannot open BG video '{BG_VIRTUAL_VIDEO_PATH}'.")
                cap_bg = None
            else:
                print(f"BG video '{BG_VIRTUAL_VIDEO_PATH}' OK.")
        else:
            print(f"Warn: BG video '{BG_VIRTUAL_VIDEO_PATH}' not found.")
            cap_bg = None
    except Exception as e:
        print(f"Error load BG video: {e}")
        cap_bg = None

    # --- LOAD EFFECTS ---
    try:
        # Effect EXCITED (fullscreen video)
        effect_excited_path = os.path.join(EFFECTS_FOLDER, EFFECT_EXCITED_VIDEO)
        if os.path.exists(effect_excited_path):
            cap_effect_excited = cv2.VideoCapture(effect_excited_path)
            if cap_effect_excited and cap_effect_excited.isOpened():
                print(f"Effect video EXCITED '{EFFECT_EXCITED_VIDEO}' OK.")
            else:
                print(f"Warn: Cannot open effect video '{EFFECT_EXCITED_VIDEO}'.")
                cap_effect_excited = None
        else:
            print(f"Warn: Effect video '{EFFECT_EXCITED_VIDEO}' not found.")
            cap_effect_excited = None
        
        # Effect LAUGH (kanan - VIDEO)
        effect_laugh_path = os.path.join(EFFECTS_FOLDER, EFFECT_LAUGH_VIDEO)
        if os.path.exists(effect_laugh_path):
            cap_effect_laugh = cv2.VideoCapture(effect_laugh_path)
            if cap_effect_laugh and cap_effect_laugh.isOpened():
                print(f"Effect video LAUGH '{EFFECT_LAUGH_VIDEO}' OK.")
            else:
                print(f"Warn: Cannot open effect video '{EFFECT_LAUGH_VIDEO}'.")
                cap_effect_laugh = None
        else:
            print(f"Warn: Effect video '{EFFECT_LAUGH_VIDEO}' not found.")
            cap_effect_laugh = None
            
        # <-- BARU: Effect THUMBS UP -->
        effect_thumbsup_path = os.path.join(EFFECTS_FOLDER, EFFECT_THUMBSUP_VIDEO)
        if os.path.exists(effect_thumbsup_path):
            cap_effect_thumbsup = cv2.VideoCapture(effect_thumbsup_path)
            if cap_effect_thumbsup and cap_effect_thumbsup.isOpened():
                print(f"Effect video THUMBSUP '{EFFECT_THUMBSUP_VIDEO}' OK.")
            else:
                print(f"Warn: Cannot open effect video '{EFFECT_THUMBSUP_VIDEO}'.")
                cap_effect_thumbsup = None
        else:
            print(f"Warn: Effect video '{EFFECT_THUMBSUP_VIDEO}' not found.")
            cap_effect_thumbsup = None
        # <-- AKHIR BLOK BARU -->
            
    except Exception as e:
        print(f"Error loading effects: {e}")
        cap_effect_excited = cap_effect_laugh = cap_effect_thumbsup = None

    # <-- BARU: Tambahkan cap_effect_thumbsup ke return -->
    return cap, stream, audio_system, holistic, cap_bg, cap_effect_excited, cap_effect_laugh, cap_effect_thumbsup, frame_h, frame_w

def initialize_state(frame_w, frame_h):
    """Inisialisasi state termasuk effect state."""
    state = {
        "frame_counter": 0,
        "stable_gesture": "NORMAL",
        "user_is_idle": True,
        "laugh_audio_counter": 0,
        "background_mode": 0,
        "resized_virtual_bg": None,
        "random_blink_timer": random.randint(RANDOM_BLINK_INTERVAL_MIN_FRAMES, RANDOM_BLINK_INTERVAL_MAX_FRAMES),
        "random_blink_counter": 0,
        "talk_frame_counter": 0,
        "gesture_buffer": deque(maxlen=GESTURE_BUFFER_SIZE),
        "current_anchor_pos": (frame_w // 2, frame_h // 2),
        "current_hair_pos": (frame_w // 2, frame_h // 2),
        "hair_velocity": (0, 0),
        "bounce_offset": (0, 0),
        "bounce_velocity": (0, 0),
        "idle_timer": random.randint(INITIAL_IDLE_DELAY_MIN_FRAMES, INITIAL_IDLE_DELAY_MAX_FRAMES),
        "idle_sequence_index": -1,
        "current_idle_offset_for_sway": (0, 0),
        "frame_count_for_process": 0,
        "last_mediapipe_results": None,
        # --- EFFECT STATE ---
        "effect_excited_active": False,
        "effect_laugh_active": False,
        "effect_thumbsup_active": False,  # <-- BARU
        "last_gesture_for_effect": "NORMAL",
        # --- LIP SYNC STATE ---
        "current_mouth_state": "closed",
        "target_mouth_state": "closed",
        "mouth_transition_progress": 0.0,
        "silence_frame_counter": 0,
        "last_rms_value": 0,
        # --- RMS MONITOR ---
        "current_rms": 0,
        "current_audio_level": "DIAM"
    }
    print("State OK.")
    return state

# --- FUNGSI LOGIKA UTAMA (Sebagian besar sama, ditambah logika effect) ---
def update_idle_state(state):
    """Update state idle."""
    if state["user_is_idle"]:
        state["idle_timer"] -= 1
        if state["idle_timer"] <= 0:
            state["idle_sequence_index"] = (state["idle_sequence_index"] + 1) % len(IDLE_SEQUENCE)
            next_s = IDLE_SEQUENCE[state["idle_sequence_index"]]
            
            if state["stable_gesture"] in ["NORMAL", "LOOK_LEFT_IDLE", "LOOK_RIGHT_IDLE", "LOOK_DOWN_IDLE"]:
                state["stable_gesture"] = next_s
            
            min_d, max_d = IDLE_STATE_DURATION_RANGES.get(next_s, (IDLE_LOOK_DURATION_MIN_FRAMES, IDLE_LOOK_DURATION_MAX_FRAMES))
            state["idle_timer"] = random.randint(min_d, max_d)
    else:
        state["idle_timer"] = random.randint(INITIAL_IDLE_DELAY_MIN_FRAMES, INITIAL_IDLE_DELAY_MAX_FRAMES)
        state["idle_sequence_index"] = -1
        state["current_idle_offset_for_sway"] = (0, 0)
        return
    
    target_o = (0, 0)
    cs = state["stable_gesture"]
    if cs == "LOOK_LEFT_IDLE":
        target_o = (-IDLE_SWAY_TARGET_OFFSET, 0)
    elif cs == "LOOK_RIGHT_IDLE":
        target_o = (IDLE_SWAY_TARGET_OFFSET, 0)
    elif cs == "LOOK_DOWN_IDLE":
        target_o = (0, IDLE_SWAY_TARGET_OFFSET)
    
    cx, cy = state["current_idle_offset_for_sway"]
    tx, ty = target_o
    nx = int(cx * (1.0 - IDLE_SWAY_SPRING_FACTOR) + tx * IDLE_SWAY_SPRING_FACTOR)
    ny = int(cy * (1.0 - IDLE_SWAY_SPRING_FACTOR) + ty * IDLE_SWAY_SPRING_FACTOR)
    
    nx = max(-30, min(30, nx))
    ny = max(-30, min(30, ny))
    
    state["current_idle_offset_for_sway"] = (nx, ny)

def process_audio(state, stream):
    """Proses audio, update state dengan lip sync detection."""
    is_talking = False
    is_laugh_detected = False
    rms = 0
    audio_level = "DIAM"
    
    if stream:
        try:
            if not stream.is_active():
                print("Warning: Audio stream tidak aktif.")
                return is_talking, is_laugh_detected, None

            available = stream.get_read_available()
            if available >= AUDIO_CHUNK_SIZE:
                audio_data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                rms = audioop.rms(audio_data, 2)
                
                # Smoothing RMS untuk transisi lebih halus
                smoothed_rms = int(state["last_rms_value"] * (1 - LIP_SYNC_SMOOTH_FACTOR) + rms * LIP_SYNC_SMOOTH_FACTOR)
                state["last_rms_value"] = smoothed_rms
                state["current_rms"] = smoothed_rms  # Untuk monitor

                # --- DETEKSI MOUTH STATE BERDASARKAN RMS ---
                if smoothed_rms < AUDIO_SPEAK_THRESHOLD_RMS:
                    # Diam
                    audio_level = "DIAM"
                    state["silence_frame_counter"] += 1
                    if state["silence_frame_counter"] > LIP_SYNC_HOLD_FRAMES:
                        state["target_mouth_state"] = "closed"
                else:
                    # Ada suara - reset silence counter
                    state["silence_frame_counter"] = 0
                    is_talking = True
                    state["user_is_idle"] = False
                    
                    # Tentukan mouth state berdasarkan volume
                    if smoothed_rms < AUDIO_RMS_LEVEL_SMALL:
                        state["target_mouth_state"] = "small"
                        audio_level = "PELAN"
                    elif smoothed_rms < AUDIO_RMS_LEVEL_MEDIUM:
                        state["target_mouth_state"] = "medium"
                        audio_level = "NORMAL"
                    elif smoothed_rms < AUDIO_RMS_LEVEL_WIDE:
                        state["target_mouth_state"] = "wide"
                        audio_level = "KERAS"
                    else:
                        # Super keras - alternate antara wide dan o untuk variasi
                        if state["frame_counter"] % 10 < 5:
                            state["target_mouth_state"] = "wide"
                        else:
                            state["target_mouth_state"] = "o"
                        audio_level = "SANGAT KERAS"

                # Deteksi tawa
                if smoothed_rms > LAUGH_RMS_THRESHOLD:
                    state["laugh_audio_counter"] = min(state["laugh_audio_counter"] + 1, LAUGH_COUNTER_MAX)
                else:
                    state["laugh_audio_counter"] = max(state["laugh_audio_counter"] - LAUGH_COUNTER_DECREMENT, 0)

                if state["laugh_audio_counter"] > LAUGH_COUNTER_THRESHOLD:
                    is_laugh_detected = True
                    is_talking = False
                    state["user_is_idle"] = False
                    audio_level = "KETAWA!"

                # Update audio level untuk monitor
                state["current_audio_level"] = audio_level

        except IOError as e:
            if e.errno == -9981:
                pass
            else:
                print(f"Warn: Audio IO Err: {e}. Disabling.")
                try:
                    if stream.is_active():
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
                stream = None
        except Exception as e:
            print(f"Audio Err: {e}")
            try:
                if stream.is_active():
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            stream = None

    return is_talking, is_laugh_detected, stream

def detect_gestures(state, results):
    """Deteksi gesture."""
    gesture = "NORMAL"
    pose_ok = results and results.pose_landmarks and hasattr(results.pose_landmarks, 'landmark')
    
    if pose_ok:
        pose = results.pose_landmarks.landmark
        try:
            if pose[mp_holistic.PoseLandmark.NOSE].visibility < 0.5:
                state["user_is_idle"] = False
        except IndexError:
            state["user_is_idle"] = False

        left_h, right_h = results.left_hand_landmarks, results.right_hand_landmarks
        left_ok, right_ok = left_h and hasattr(left_h, 'landmark'), right_h and hasattr(right_h, 'landmark')

        try:
            if left_ok and right_ok:
                lw = pose[mp_holistic.PoseLandmark.LEFT_WRIST]
                rw = pose[mp_holistic.PoseLandmark.RIGHT_WRIST]
                n = pose[mp_holistic.PoseLandmark.NOSE]
                if lw.visibility > 0.5 and rw.visibility > 0.5 and n.visibility > 0.5 and lw.y < n.y and rw.y < n.y and abs(lw.x - rw.x) < GESTURE_BLUSH_WRIST_DISTANCE_THRESHOLD:
                    gesture = "BLUSHING"

            if gesture == "NORMAL":
                rw_lm = pose[mp_holistic.PoseLandmark.RIGHT_WRIST]
                rs_lm = pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                lw_lm = pose[mp_holistic.PoseLandmark.LEFT_WRIST]
                ls_lm = pose[mp_holistic.PoseLandmark.LEFT_SHOULDER]

                r_wave = right_ok and rw_lm.visibility > 0.5 and rs_lm.visibility > 0.5 and rw_lm.y < rs_lm.y
                l_wave = left_ok and lw_lm.visibility > 0.5 and ls_lm.visibility > 0.5 and lw_lm.y < ls_lm.y
                
                if r_wave and l_wave:
                    gesture = "EXCITED"
                elif r_wave or l_wave:
                    gesture = "WAVE"
                    
                elif right_ok:
                    finger_status = get_finger_status(right_h)
                    
                    try:
                        thumb_tip = right_h.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index_tip = right_h.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        
                        distance = math.hypot(
                            index_tip.x - thumb_tip.x,
                            index_tip.y - thumb_tip.y
                        )
                        
                        if distance < 0.10:
                            gesture = "OK"
                        
                        elif finger_status['THUMB'] and not finger_status['INDEX'] and not finger_status['MIDDLE']:
                            gesture = "THUMBS_UP"
                        
                        elif finger_status['INDEX'] and finger_status['MIDDLE'] and not finger_status['RING']:
                            gesture = "PEACE"
                            
                    except (IndexError, AttributeError) as e:
                        print(f"  [ERROR] Error saat cek gesture jari: {e}")

        except IndexError:
            pass
        except Exception as e:
            print(f"Error saat deteksi gesture (umum): {e}")
    else:
        state["user_is_idle"] = False

    state["gesture_buffer"].append(gesture)
    final_gesture = "NORMAL"
    
    if len(state["gesture_buffer"]) == GESTURE_BUFFER_SIZE:
        mcg = max(set(state["gesture_buffer"]), key=state["gesture_buffer"].count)
        count = state["gesture_buffer"].count(mcg)
        
        if count >= GESTURE_CONFIRMATION_THRESHOLD and mcg != "NORMAL":
            final_gesture = mcg
            state["user_is_idle"] = False
        elif state["gesture_buffer"].count("NORMAL") < (GESTURE_BUFFER_SIZE - GESTURE_CONFIRMATION_THRESHOLD + 1):
            state["user_is_idle"] = False

    else:
        state["user_is_idle"] = False

    if not state["user_is_idle"]:
        cs = state["stable_gesture"]
        is_idle = cs in ["NORMAL", "LOOK_LEFT_IDLE", "LOOK_RIGHT_IDLE", "LOOK_DOWN_IDLE"]
        new_g = final_gesture if final_gesture != "NORMAL" else "NORMAL"
        if new_g != cs or not is_idle:
            state["stable_gesture"] = new_g
            
            state["idle_timer"] = random.randint(INITIAL_IDLE_DELAY_MIN_FRAMES, INITIAL_IDLE_DELAY_MAX_FRAMES)
            state["idle_sequence_index"] = -1

def determine_blink_state(state, results):
    """Deteksi kedip."""
    is_blinking = False
    if results and results.face_landmarks:
        face_lm = results.face_landmarks
        le = get_aspect_ratio(face_lm, 386, 374, 362, 263)
        re = get_aspect_ratio(face_lm, 159, 145, 133, 33)
        if le > 1e-6 and re > 1e-6 and (le + re) / 2.0 < EYE_AR_THRESHOLD:
            is_blinking = True
            state["random_blink_timer"] = random.randint(RANDOM_BLINK_INTERVAL_MIN_FRAMES, RANDOM_BLINK_INTERVAL_MAX_FRAMES)
            state["random_blink_counter"] = 0
    force_blink = False
    if state["random_blink_counter"] > 0:
        force_blink = True
        state["random_blink_counter"] -= 1
    else:
        state["random_blink_timer"] -= 1
    if state["random_blink_timer"] <= 0:
        force_blink = True
        state["random_blink_counter"] = RANDOM_BLINK_DURATION_FRAMES
        state["random_blink_timer"] = random.randint(RANDOM_BLINK_INTERVAL_MIN_FRAMES, RANDOM_BLINK_INTERVAL_MAX_FRAMES)
    return is_blinking, force_blink

def select_assets(state, assets, is_blinking, force_blink, is_talking, is_laugh_detected):
    """Pilih aset kepala & badan dengan advanced lip sync."""
    sg = state["stable_gesture"]
    if sg not in assets['gesture_config']:
        sg = "NORMAL"
        state["stable_gesture"] = "NORMAL"
    if is_laugh_detected:
        sg = "LAUGH"
        state["random_blink_counter"] = 0
        state["talk_frame_counter"] = 0

    cfg = assets['gesture_config'].get(sg, assets['gesture_config']["NORMAL"])
    body_img, base_head = cfg.get("body"), cfg.get("head")
    if base_head is None or body_img is None:
        print(f"FATAL: Missing base assets '{sg}'. Using NORMAL.")
        cfg = assets['gesture_config']["NORMAL"]
        body_img, base_head = cfg.get("body"), cfg.get("head")
    if base_head is None or body_img is None:
        print("FATAL: Missing NORMAL assets!")
        return None, None

    final_head = base_head
    variations = assets['head_variations'].get(id(base_head))

    # --- PRIORITAS ANIMASI ---
    # 1. Laugh (tertinggi)
    if is_laugh_detected:
        final_head = assets.get("head_laugh", base_head)
    # 2. Excited
    elif sg == "EXCITED":
        final_head = assets.get("head_excited", base_head)
    # 3. Blink
    elif (is_blinking or force_blink):
        blink = variations.get("blink") if variations else None
        if blink is not None:
            final_head = blink
        else:
            fallback_blink = assets.get("head_blush_blink") if sg == "BLUSHING" and assets.get("head_blush_blink") is not None else assets.get("head_normal_blink")
            if fallback_blink is not None:
                final_head = fallback_blink
    # 4. LIP SYNC (hanya untuk gesture NORMAL)
    elif is_talking and sg == "NORMAL" and variations:
        # Dapatkan mouth state dari audio processing
        target_state = state.get("target_mouth_state", "closed")
        
        # Ambil mouth variation
        mouth_img = variations.get(target_state)
        if mouth_img is not None:
            final_head = mouth_img
            state["current_mouth_state"] = target_state
        else:
            # Fallback ke closed jika mouth variation tidak ada
            closed_img = variations.get("closed")
            if closed_img is not None:
                final_head = closed_img
            else:
                final_head = base_head
    # 5. Closed mouth (default saat diam)
    elif not is_talking and sg == "NORMAL" and variations:
        closed_img = variations.get("closed")
        if closed_img is not None:
            final_head = closed_img
            state["current_mouth_state"] = "closed"

    # Fallback final
    if final_head is None:
        final_head = assets.get("head_normal")
    if body_img is None:
        body_img = assets.get("body_normal")
    if final_head is None or body_img is None:
        print("FATAL: Fallback final failed.")
        return None, None

    return final_head, body_img

def calculate_positions(state, results, frame_w, frame_h, current_head_img, current_body_img, current_hair_img):
    """Hitung posisi."""
    target_pos = (frame_w // 2, frame_h // 2)
    if results and results.pose_landmarks:
        try:
            n = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
            if n.visibility > 0.5:
                target_pos = (int(n.x * frame_w), int(n.y * frame_h))
        except (IndexError, AttributeError):
            pass

    cx, cy = state["current_anchor_pos"]
    tx, ty = target_pos
    ax = int(cx * (1.0 - POSE_SPRING_FACTOR) + tx * POSE_SPRING_FACTOR)
    ay = int(cy * (1.0 - POSE_SPRING_FACTOR) + ty * POSE_SPRING_FACTOR)
    state["current_anchor_pos"] = (ax, ay)

    target_hair_x = ax + HAIR_OFFSET_X
    target_hair_y = ay + HAIR_OFFSET_Y
    
    chx, chy = state["current_hair_pos"]
    
    force_x = (target_hair_x - chx) * HAIR_SPRING_FACTOR
    force_y = (target_hair_y - chy) * HAIR_SPRING_FACTOR
    
    vel_x, vel_y = state.get("hair_velocity", (0, 0))
    vel_x = (vel_x + force_x) * HAIR_DRAG_FACTOR
    vel_y = (vel_y + force_y) * HAIR_DRAG_FACTOR
    
    new_hair_x = chx + vel_x
    new_hair_y = chy + vel_y
    
    dx = new_hair_x - target_hair_x
    dy = new_hair_y - target_hair_y
    distance = math.sqrt(dx * dx + dy * dy)
    if distance > HAIR_MAX_DISTANCE:
        scale = HAIR_MAX_DISTANCE / distance
        new_hair_x = target_hair_x + dx * scale
        new_hair_y = target_hair_y + dy * scale
        vel_x *= 0.5
        vel_y *= 0.5
    
    state["current_hair_pos"] = (int(new_hair_x), int(new_hair_y))
    state["hair_velocity"] = (vel_x, vel_y)
    
    hax, hay = int(new_hair_x), int(new_hair_y)

    hh, hw = current_head_img.shape[:2] if current_head_img is not None else (100, 100)
    bh, bw = current_body_img.shape[:2] if current_body_img is not None else (200, 150)
    hrh, hrw = current_hair_img.shape[:2] if current_hair_img is not None else (150, 150)

    pxh = ax - (hw // 2) + HEAD_GLOBAL_OFFSET[0]
    pyh = ay - (hh // 2) + HEAD_GLOBAL_OFFSET[1]
    pxb = ax - (bw // 2) + BODY_OFFSET_FROM_HEAD_X
    pyb = pyh + BODY_OFFSET_FROM_HEAD_Y
    pxhr = hax - (hrw // 2)
    pyhr = hay - (hrh // 2)

    fc = state["frame_counter"]
    swy = int(math.sin(fc * BREATH_SWAY_Y_FREQ) * BREATH_SWAY_Y_AMP)
    swx = int(math.cos(fc * BREATH_SWAY_X_FREQ) * BREATH_SWAY_X_AMP)
    iswx, iswy = state["current_idle_offset_for_sway"]

    iswx = max(-20, min(20, iswx))
    iswy = max(-20, min(20, iswy))
    
    tswx = iswx + swx
    tswy = iswy + swy
    
    tswx = max(-30, min(30, tswx))
    tswy = max(-30, min(30, tswy))
    
    pyh += tswy
    pyb += tswy
    pyhr += tswy
    pxh += tswx
    pxb += tswx
    pxhr += tswx

    if state["stable_gesture"] == "LAUGH":
        shx = int(random.uniform(-LAUGH_SHAKE_AMOUNT_X, LAUGH_SHAKE_AMOUNT_X))
        shy = int(random.uniform(-LAUGH_SHAKE_AMOUNT_Y, LAUGH_SHAKE_AMOUNT_Y))
        pyh += shy
        pyb += shy
        pyhr += shy
        pxh += shx
        pxb += shx
        pxhr += shx

    return int(pxh), int(pyh), int(pxb), int(pyb), int(pxhr), int(pyhr)

def draw_background(state, frame_w, frame_h, cap_bg, bg_virtual_img):
    """Gambar background."""
    bg_mode = state["background_mode"]
    if frame_h <= 0 or frame_w <= 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    fallback = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    fallback[:] = BG_FALLBACK_COLOR
    if bg_mode == 0:
        if bg_virtual_img is None:
            return fallback
        cache = state.get("resized_virtual_bg")
        if cache is not None and cache.shape[0] == frame_h and cache.shape[1] == frame_w:
            return cache.copy()
        try:
            if bg_virtual_img.shape[0] <= 0 or bg_virtual_img.shape[1] <= 0:
                return fallback
            state["resized_virtual_bg"] = cv2.resize(bg_virtual_img, (frame_w, frame_h))
            return state["resized_virtual_bg"].copy()
        except Exception as e:
            print(f"Err resize BG img: {e}")
            state["resized_virtual_bg"] = None
            return fallback
    elif bg_mode == 1:
        if cap_bg is None or not cap_bg.isOpened():
            return fallback
        ret, frame = cap_bg.read()
        if not ret:
            cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap_bg.read()
        if ret and frame is not None:
            try:
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    return cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
                else:
                    return fallback
            except Exception as e:
                print(f"Err resize BG video: {e}")
                return fallback
        else:
            return fallback
    elif bg_mode == 2:
        bg = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        bg[:] = BG_GREEN_SCREEN_COLOR
        return bg
    else:
        return fallback

def draw_avatar_and_ui(final_output, state, assets, current_head_img, current_body_img,
                         pos_x_head, pos_y_head, pos_x_body, pos_y_body,
                         pos_x_hair, pos_y_hair):
    """Gambar rambut, avatar, UI."""
    temp_output = final_output.copy()
    hair_img = assets.get("hair_back")
    if hair_img is not None:
        temp_output = overlay_png(temp_output, hair_img, pos_x_hair, pos_y_hair)
    body_drawn, head_drawn = False, False
    if state["stable_gesture"] == "BLUSHING":
        if current_head_img is not None:
            temp_output = overlay_png(temp_output, current_head_img, pos_x_head, pos_y_head)
            head_drawn = True
        if current_body_img is not None:
            temp_output = overlay_png(temp_output, current_body_img, pos_x_body, pos_y_body)
            body_drawn = True
    else:
        if current_body_img is not None:
            temp_output = overlay_png(temp_output, current_body_img, pos_x_body, pos_y_body)
            body_drawn = True
        if current_head_img is not None:
            temp_output = overlay_png(temp_output, current_head_img, pos_x_head, pos_y_head)
            head_drawn = True
    if temp_output is not None:
        final_output = temp_output
    
    # UI Text
    bg_idx = state["background_mode"]
    mode_txt = BG_MODES[bg_idx] if 0 <= bg_idx < len(BG_MODES) else "BG:?"
    final_output = draw_fancy_text(final_output, mode_txt, UI_MODE_POS, UI_FONT, UI_MODE_SCALE, UI_TEXT_COLOR, UI_MODE_THICKNESS, UI_BG_COLOR, UI_BG_ALPHA, UI_PADDING)
    
    exit_keys = [f"'{chr(k)}'" if k < 256 else '?' for k in EXIT_KEYS if k != 27] + (["'Esc'"] if 27 in EXIT_KEYS else [])
    exit_txt = "/".join(exit_keys) if exit_keys else "??"
    help_txt = f"'{chr(BG_CHANGE_KEY)}': BG | {exit_txt}: Keluar"
    final_output = draw_fancy_text(final_output, help_txt, UI_HELP_POS, UI_FONT, UI_HELP_SCALE, UI_TEXT_COLOR, UI_HELP_THICKNESS, UI_BG_COLOR, UI_BG_ALPHA, UI_PADDING)
    
    # RMS Monitor
    if UI_RMS_MONITOR_ENABLED:
        rms_val = state.get("current_rms", 0)
        audio_level = state.get("current_audio_level", "DIAM")
        laugh_counter = state.get("laugh_audio_counter", 0)
        
        # Warna berdasarkan level
        if audio_level == "KETAWA!":
            color = (0, 0, 255)  # Merah untuk tawa
        elif audio_level == "SANGAT KERAS":
            color = (0, 165, 255)  # Orange
        elif audio_level == "KERAS":
            color = (0, 255, 255)  # Kuning
        elif audio_level == "NORMAL":
            color = (0, 255, 0)  # Hijau
        elif audio_level == "PELAN":
            color = (255, 255, 0)  # Cyan
        else:
            color = (255, 255, 255)  # Putih untuk diam
        
        monitor_txt = f"RMS: {rms_val:3d} | Level: {audio_level:12s} | Laugh: {laugh_counter:2d}/{LAUGH_COUNTER_THRESHOLD}"
        final_output = draw_fancy_text(final_output, monitor_txt, UI_RMS_POS, UI_FONT, UI_RMS_SCALE, color, UI_RMS_THICKNESS, UI_BG_COLOR, UI_BG_ALPHA, UI_PADDING)
    
    return final_output

def main():
    assets = None
    cap = stream = audio_system = holistic = cap_bg = None
    cap_effect_excited = cap_effect_laugh = cap_effect_thumbsup = None  # <-- BARU
    
    try:
        assets = load_assets(ASSET_FOLDER)
        
        # <-- BARU: Menambahkan cap_effect_thumbsup saat unpacking -->
        cap, stream, audio_system, holistic, cap_bg, cap_effect_excited, cap_effect_laugh, cap_effect_thumbsup, frame_h, frame_w = initialize_systems()
        
        state = initialize_state(frame_w, frame_h)

        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Info: End of video/camera.")
                    break
                if len(frame.shape) < 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
                    print("Warn: Invalid frame.")
                    time.sleep(0.1)
                    continue
                frame = cv2.flip(frame, 1)
                state["frame_counter"] += 1
                new_h, new_w, _ = frame.shape
                if new_h != frame_h or new_w != frame_w:
                    print(f"Frame resize: {frame_w}x{frame_h} -> {new_w}x{new_h}")
                    frame_h, frame_w = new_h, new_w
                    state["resized_virtual_bg"] = None
            except Exception as e:
                print(f"Error reading frame: {e}")
                break

            state["user_is_idle"] = True
            is_talking, is_laugh_detected, stream = process_audio(state, stream)

            results = None
            state["frame_count_for_process"] += 1
            if state["frame_count_for_process"] >= FRAME_PROCESS_INTERVAL:
                state["frame_count_for_process"] = 0
                if frame is not None and frame.size > 0:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    try:
                        results = holistic.process(image_rgb)
                    finally:
                        image_rgb.flags.writeable = True
                    if results and (results.pose_landmarks or results.face_landmarks or results.left_hand_landmarks or results.right_hand_landmarks):
                        state["last_mediapipe_results"] = results
                    else:
                        state["last_mediapipe_results"] = None
                        state["user_is_idle"] = False
                else:
                    results = state.get("last_mediapipe_results", None)
            else:
                results = state.get("last_mediapipe_results", None)
            if results is None:
                state["user_is_idle"] = False

            is_blinking, force_blink = determine_blink_state(state, results)
            detect_gestures(state, results)
            update_idle_state(state)

            if is_laugh_detected:
                state["stable_gesture"] = "LAUGH"
                
            # --- UPDATE EFFECT STATE ---
            current_gesture = state["stable_gesture"]
            last_gesture = state["last_gesture_for_effect"]
            
            # Trigger effect saat gesture berubah
            if current_gesture != last_gesture:
                # EXCITED effect (fullscreen video)
                if current_gesture == "EXCITED":
                    state["effect_excited_active"] = True
                    if cap_effect_excited: cap_effect_excited.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    state["effect_excited_active"] = False
                
                # LAUGH effect
                if current_gesture == "LAUGH":
                    state["effect_laugh_active"] = True
                    if cap_effect_laugh: cap_effect_laugh.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    state["effect_laugh_active"] = False
                
                # <-- BARU: THUMBS_UP effect -->
                if current_gesture == "THUMBS_UP":
                    state["effect_thumbsup_active"] = True
                    if cap_effect_thumbsup: cap_effect_thumbsup.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    state["effect_thumbsup_active"] = False
                # <-- AKHIR BLOK BARU -->
                
                state["last_gesture_for_effect"] = current_gesture
            
            # Update effect status berdasarkan is_laugh_detected dari audio
            if is_laugh_detected and current_gesture == "LAUGH":
                if not state["effect_laugh_active"]:
                    state["effect_laugh_active"] = True
                    if cap_effect_laugh: cap_effect_laugh.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("[DEBUG] LAUGH effect RE-activated from audio!")  # Debug message

            current_head_img, current_body_img = select_assets(state, assets, is_blinking, force_blink, is_talking, is_laugh_detected)
            if current_head_img is None or current_body_img is None:
                print("FATAL: Asset selection failed.")
                break

            pos_x_head, pos_y_head, pos_x_body, pos_y_body, pos_x_hair, pos_y_hair = calculate_positions(
                state, results, frame_w, frame_h, current_head_img, current_body_img, assets.get("hair_back")
            )

            final_output = draw_background(state, frame_w, frame_h, cap_bg, assets.get("bg_virtual_img"))
            
            # --- APPLY VIDEO EFFECTS SEBELUM AVATAR ---
            
            # Effect EXCITED (fullscreen, di background layer)
            if state["effect_excited_active"] and cap_effect_excited and cap_effect_excited.isOpened():
                ret_fx, frame_fx = cap_effect_excited.read()
                if ret_fx and frame_fx is not None:
                    final_output = overlay_video_effect(final_output, frame_fx, 'fullscreen', frame_w, frame_h)
                else:
                    # Video selesai, matikan effect
                    state["effect_excited_active"] = False
            
            # <-- DIUBAH: Effect LAUGH (fullscreen, di background layer) -->
            if state["effect_laugh_active"] and cap_effect_laugh and cap_effect_laugh.isOpened():
                ret_fx, frame_fx = cap_effect_laugh.read()
                if ret_fx and frame_fx is not None:
                    final_output = overlay_video_effect(final_output, frame_fx, 'fullscreen', frame_w, frame_h)
                else:
                    # Video selesai, loop kembali
                    cap_effect_laugh.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # <-- BARU: Effect THUMBS_UP (fullscreen, di background layer) -->
            if state["effect_thumbsup_active"] and cap_effect_thumbsup and cap_effect_thumbsup.isOpened():
                ret_fx, frame_fx = cap_effect_thumbsup.read()
                if ret_fx and frame_fx is not None:
                    final_output = overlay_video_effect(final_output, frame_fx, 'fullscreen', frame_w, frame_h)
                else:
                    # Video selesai, matikan effect
                    state["effect_thumbsup_active"] = False
            
            # Gambar avatar
            final_output = draw_avatar_and_ui(final_output, state, assets, current_head_img, current_body_img,
                                              pos_x_head, pos_y_head, pos_x_body, pos_y_body,
                                              pos_x_hair, pos_y_hair)
            
            # --- APPLY EFFECTS SETELAH AVATAR ---
            # (Blok effect laugh sudah dipindah ke atas)
            
            try:
                cv2.imshow(WINDOW_NAME, final_output)
            except Exception as e:
                print(f"Error showing frame: {e}")
                break

            key = cv2.waitKey(5) & 0xFF
            if key in EXIT_KEYS:
                print("Exit key.")
                break
            elif key == BG_CHANGE_KEY:
                state["background_mode"] = (state["background_mode"] + 1) % len(BG_MODES)
                state["resized_virtual_bg"] = None

    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Asset load failed: {e}")
    except IOError as e:
        print(f"CRITICAL ERROR: IO problem (camera/file): {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        resources = {
            "Camera": cap,
            "BG Video": cap_bg,
            "Effect EXCITED": cap_effect_excited,
            "Effect LAUGH": cap_effect_laugh,
            "Effect THUMBSUP": cap_effect_thumbsup,  # <-- BARU
            "Audio Stream": stream,
            "Audio System": audio_system,
            "MediaPipe": holistic
        }
        for name, res in resources.items():
            try:
                if res is not None:
                    if isinstance(res, cv2.VideoCapture) and res.isOpened():
                        res.release()
                    elif isinstance(res, pyaudio.Stream) and hasattr(res, 'is_active') and res.is_active():
                        res.stop_stream()
                        res.close()
                    elif isinstance(res, pyaudio.PyAudio):
                        res.terminate()
                    elif hasattr(res, 'close'):
                        res.close()
                    print(f" - {name} cleaned.")
            except Exception as e:
                print(f" Error cleaning {name}: {e}")
        try:
            cv2.destroyAllWindows()
            print(" - Windows closed.")
        except Exception as e:
            print(f" Error closing windows: {e}")
        print("Program finished.")

if __name__ == "__main__":
    main()