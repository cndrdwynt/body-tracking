# config.py
# File ini berisi semua konstanta dan pengaturan.

import numpy as np
import os

# --- Folder Paths ---
IMAGE_FOLDER = 'images'
BACKGROUND_FOLDER = 'backgrounds'
EFFECTS_FOLDER = 'effects'

# --- KONSTANTA EFFECTS ---
EFFECT_EXCITED_VIDEO = 'excited_effect.mp4'
EFFECT_LAUGH_VIDEO = 'laugh_effect.mp4'
EFFECT_THUMBSUP_VIDEO = 'thumbsup_effect.mp4'

# Pengaturan Chroma Key (Greenscreen removal)
CHROMA_KEY_COLOR_LOWER = np.array([40, 40, 40])
CHROMA_KEY_COLOR_UPPER = np.array([80, 255, 255])
CHROMA_TOLERANCE = 30

# --- KONSTANTA UTAMA ---
WEBCAM_INDEX = 0
WINDOW_NAME = 'VTuber Margo Interaktif (OpenCV Project)'
EXIT_KEYS = [ord('q'), 27]
BG_CHANGE_KEY = ord('b')

HEAD_SCALE_FACTOR = 0.4
BODY_SCALE_FACTOR = 0.5
HEAD_GLOBAL_OFFSET = (5, 80)
BODY_OFFSET_FROM_HEAD_X = 0
BODY_OFFSET_FROM_HEAD_Y = -90

MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5
FRAME_PROCESS_INTERVAL = 1

# --- KONSTANTA FISIKA ---
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
AUDIO_FORMAT = 16 # pyaudio.paInt16 adalah 16

# Audio RMS Thresholds untuk deteksi suara
AUDIO_SPEAK_THRESHOLD_RMS = 2500
AUDIO_RMS_LEVEL_SMALL = 3000
AUDIO_RMS_LEVEL_MEDIUM = 4000
AUDIO_RMS_LEVEL_WIDE = 4500
AUDIO_RMS_LEVEL_O = 6000

# --- KONSTANTA LIP SYNC VISUAL (BARU!) ---
# Landmark indices MediaPipe Face Mesh
MOUTH_LANDMARK_UPPER = 13      # Upper lip center
MOUTH_LANDMARK_LOWER = 14      # Lower lip center
MOUTH_LANDMARK_LEFT = 61       # Left mouth corner
MOUTH_LANDMARK_RIGHT = 291     # Right mouth corner

# Thresholds untuk mouth openness ratio
MOUTH_OPENNESS_THRESHOLD_CLOSED = 0.15   # < 0.15 = closed

MOUTH_OPENNESS_THRESHOLD_SMALL = 0.30    # 0.15 - 0.30
MOUTH_OPENNESS_THRESHOLD_MEDIUM = 0.40   # 0.30 - 0.40
MOUTH_OPENNESS_THRESHOLD_WIDE = 0.48     # 0.40 - 0.55
MOUTH_OPENNESS_THRESHOLD_O = 0.90

# Mouth width thresholds (untuk deteksi bentuk O vs Wide)
MOUTH_WIDTH_RATIO_THRESHOLD_O = 0.85     # Jika width < 0.85 dari normal = bentuk O

# --- HYBRID LIP SYNC MODE ---
# Mode: "AUDIO", "VISUAL", "HYBRID"
LIP_SYNC_MODE = "VISUAL"

# Bobot untuk mode HYBRID (total harus = 1.0)
HYBRID_AUDIO_WEIGHT = 0.25
HYBRID_VISUAL_WEIGHT = 0.75

# Confidence thresholds
VISUAL_CONFIDENCE_THRESHOLD = 0.6
AUDIO_CONFIDENCE_THRESHOLD = 0.3

# Smoothing
LIP_SYNC_SMOOTH_FACTOR = 0.4
MOUTH_TRANSITION_SPEED = 0.35
LIP_SYNC_HOLD_FRAMES = 2

# --- DETEKSI TAWA (IMPROVED!) ---
# Audio-based laugh detection
LAUGH_RMS_THRESHOLD = 18000
LAUGH_COUNTER_THRESHOLD = 4
LAUGH_COUNTER_MAX = 20
LAUGH_COUNTER_DECREMENT = 1

# Visual-based laugh detection (BARU!)
LAUGH_MOUTH_OPENNESS_THRESHOLD = 0.60   # Mulut sangat lebar
LAUGH_DURATION_MIN_FRAMES = 2          # Minimal durasi mulut terbuka lebar
LAUGH_COOLDOWN_FRAMES = 0             # Cooldown sebelum bisa laugh lagi

# Combined laugh detection
LAUGH_AUDIO_WEIGHT = 0.0               # <-- Diubah ke 0.0
LAUGH_VISUAL_WEIGHT = 1.0              # <-- Diubah ke 1.0
LAUGH_COMBINED_THRESHOLD = 0.55        # Threshold untuk visual

# Laugh shake effect
LAUGH_SHAKE_AMOUNT_X = 15
LAUGH_SHAKE_AMOUNT_Y = 10

# --- KONSTANTA DETEKSI ---
EYE_AR_THRESHOLD = 0.3
RANDOM_BLINK_DURATION_FRAMES = 6
RANDOM_BLINK_INTERVAL_MIN_FRAMES = 60
RANDOM_BLINK_INTERVAL_MAX_FRAMES = 150

GESTURE_BUFFER_SIZE = 6
GESTURE_CONFIRMATION_THRESHOLD = 2
GESTURE_OK_DISTANCE_THRESHOLD = 0.20
GESTURE_BLUSH_WRIST_DISTANCE_THRESHOLD = 0.3
FACE_DIRECTION_THRESHOLD = 0.05 # Perbedaan koordinat X hidung vs bahu untuk deteksi tengok

# --- KONSTANTA IDLE (BODY) ---
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

# --- KONSTANTA UI ---
UI_FONT = 0 # cv2.FONT_HERSHEY_DUPLEX
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
UI_RMS_MONITOR_ENABLED = False # <-- SUDAH DISET KE FALSE
UI_RMS_POS = (10, 90)
UI_RMS_SCALE = 0.5
UI_RMS_THICKNESS = 1

# --- KONSTANTA BACKGROUND ---
BG_VIRTUAL_IMAGE_PATH = 'background_virtual.jpeg'
BG_VIRTUAL_VIDEO_PATH = 'background_video.mp4'
BG_MODES = ['BG: VIRTUAL IMAGE', 'BG: VIRTUAL VIDEO', 'BG: GREEN SCREEN']
BG_FALLBACK_COLOR = (20, 20, 20)
BG_GREEN_SCREEN_COLOR = (0, 255, 0)

# --- KONFIGURASI ASET ---
ASSET_FILES = {
    "head_normal": 'margo_head_normal.png',
    "head_blush": 'margo_head_blush.png',
    "head_excited": 'margo_head_excited.png',
    "head_normal_blink": 'margo_head_normal_blink.png',
    "head_normal_closed": 'margo_head_normal.png',
    "head_normal_small": 'margo_mouth_small.png',
    "head_normal_medium": 'margo_mouth_medium.png',
    "head_normal_wide": 'margo_mouth_wide.png',
    "head_normal_o": 'margo_mouth_o.png',
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
}

HEAD_VARIATION_MAP_CONFIG = {
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

# --- EYE OVERLAY SYSTEM ---
EYE_OVERLAY_FILES = {
    "eyes_normal": "eyes_normal.png",        # Mata normal (terbuka, tengah)
    "eyes_look_left": "eyes_look_left.png",  # Mata lihat kiri
    "eyes_look_right": "eyes_look_right.png" # Mata lihat kanan
}

# Posisi mata relatif terhadap head (offset dari head position)
# Sesuaikan nilai ini agar mata overlay pas dengan wajah karakter
EYE_OFFSET_X = -7      # Geser horizontal (positif = kanan, negatif = kiri)
EYE_OFFSET_Y = -55      # Geser vertikal (positif = bawah, negatif = atas)

# --- EYE IDLE ANIMATION SYSTEM ---
# Durasi idle sebelum mulai lihat kiri-kanan (dalam frames)
# Asumsi 30 FPS: 1 detik = 30 frames
EYE_IDLE_DELAY_MIN_FRAMES = 180  # 6 detik sebelum mulai lihat-lihat
EYE_IDLE_DELAY_MAX_FRAMES = 300  # 10 detik maksimal delay

# Durasi melihat ke satu arah (dalam frames)
EYE_LOOK_DURATION_MIN_FRAMES = 20   # 0.7 detik (Sebelumnya 45)
EYE_LOOK_DURATION_MAX_FRAMES = 40   # 1.3 detik (Sebelumnya 90)

# Durasi kembali ke normal sebelum lihat arah lain
EYE_RETURN_TO_CENTER_FRAMES = 10    # 0.3 detik (Sebelumnya 15)

# Sequence eye idle animation
# Urutan: tengah → kiri → tengah → kanan → tengah → repeat
EYE_IDLE_SEQUENCE = ["CENTER", "LEFT", "CENTER", "RIGHT", "CENTER"]

GESTURE_CONFIG_SETUP = [
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