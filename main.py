# main.py
# File utama untuk menjalankan aplikasi VTuber dengan DEBUG MOUTH DETECTION

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

# --- IMPORT FILE LOKAL KITA ---
import config as cfg
from utils import *
from vtuber_core import *
from renderer import *

# ... (Fungsi initialize_systems() tidak berubah) ...
try:
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands
except AttributeError:
    print("Error: Gagal init mediapipe.")
    exit()

# --- FUNGSI INISIALISASI ---

def initialize_systems():
    """Inisialisasi sistem termasuk video effect."""
    cap = stream = audio_system = holistic = cap_bg = None
    cap_effect_excited = cap_effect_laugh = cap_effect_thumbsup = None
    frame_h, frame_w = 480, 640

    # Kamera
    try:
        cap = cv2.VideoCapture(cfg.WEBCAM_INDEX)
        if not cap.isOpened():
            raise IOError(f"Cannot open webcam {cfg.WEBCAM_INDEX}")
        ret, frame = cap.read()
        if not ret or frame is None:
            raise IOError("Cannot read initial frame.")
        frame_h, frame_w, _ = frame.shape
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"Kamera {cfg.WEBCAM_INDEX} ({frame_w}x{frame_h}) OK.")
    except Exception as e:
        print(f"Error init kamera: {e}")
        if cap: cap.release()
        raise

    # Audio
    try:
        audio_system = pyaudio.PyAudio()
        stream = audio_system.open(format=cfg.AUDIO_FORMAT, channels=cfg.AUDIO_CHANNELS,
                                      rate=cfg.AUDIO_SAMPLE_RATE, input=True,
                                      frames_per_buffer=cfg.AUDIO_CHUNK_SIZE, start=True)
        print("Mikrofon OK.")
    except Exception as e:
        print(f"Warn: Gagal init audio: {e}. No audio features.")
        if stream: stream.close()
        if audio_system: audio_system.terminate()
        audio_system = stream = None

    # MediaPipe
    try:
        holistic = mp_holistic.Holistic(min_detection_confidence=cfg.MP_MIN_DETECTION_CONFIDENCE,
                                        min_tracking_confidence=cfg.MP_MIN_TRACKING_CONFIDENCE,
                                        enable_segmentation=True)
        print("MediaPipe OK.")
    except Exception as e:
        print(f"Error init MP: {e}")
        if cap: cap.release()
        if stream: stream.close()
        if audio_system: audio_system.terminate()
        raise

    # BG Video
    try:
        # Path sekarang menggunakan BACKGROUND_FOLDER dari config
        path = os.path.join(cfg.BACKGROUND_FOLDER, cfg.BG_VIRTUAL_VIDEO_PATH)
        if os.path.exists(path):
            cap_bg = cv2.VideoCapture(path)
            if cap_bg is None or not cap_bg.isOpened():
                print(f"Warn: Cannot open BG video '{path}'.")
                cap_bg = None
            else:
                print(f"BG video '{path}' OK.")
        else:
            print(f"Warn: BG video '{path}' not found.")
            cap_bg = None
    except Exception as e:
        print(f"Error load BG video: {e}")
        cap_bg = None

    # --- LOAD EFFECTS ---
    try:
        # Effect EXCITED
        effect_excited_path = os.path.join(cfg.EFFECTS_FOLDER, cfg.EFFECT_EXCITED_VIDEO)
        if os.path.exists(effect_excited_path):
            cap_effect_excited = cv2.VideoCapture(effect_excited_path)
            if cap_effect_excited and cap_effect_excited.isOpened():
                print(f"Effect video EXCITED '{effect_excited_path}' OK.")
            else:
                print(f"Warn: Cannot open effect video '{effect_excited_path}'.")
                cap_effect_excited = None
        else:
            print(f"Warn: Effect video '{effect_excited_path}' not found.")
            cap_effect_excited = None
        
        # Effect LAUGH
        effect_laugh_path = os.path.join(cfg.EFFECTS_FOLDER, cfg.EFFECT_LAUGH_VIDEO)
        if os.path.exists(effect_laugh_path):
            cap_effect_laugh = cv2.VideoCapture(effect_laugh_path)
            if cap_effect_laugh and cap_effect_laugh.isOpened():
                print(f"Effect video LAUGH '{effect_laugh_path}' OK.")
            else:
                print(f"Warn: Cannot open effect video '{effect_laugh_path}'.")
                cap_effect_laugh = None
        else:
            print(f"Warn: Effect video '{effect_laugh_path}' not found.")
            cap_effect_laugh = None
            
        # Effect THUMBS UP
        effect_thumbsup_path = os.path.join(cfg.EFFECTS_FOLDER, cfg.EFFECT_THUMBSUP_VIDEO)
        if os.path.exists(effect_thumbsup_path):
            cap_effect_thumbsup = cv2.VideoCapture(effect_thumbsup_path)
            if cap_effect_thumbsup and cap_effect_thumbsup.isOpened():
                print(f"Effect video THUMBSUP '{effect_thumbsup_path}' OK.")
            else:
                print(f"Warn: Cannot open effect video '{effect_thumbsup_path}'.")
                cap_effect_thumbsup = None
        else:
            print(f"Warn: Effect video '{effect_thumbsup_path}' not found.")
            cap_effect_thumbsup = None
            
    except Exception as e:
        print(f"Error loading effects: {e}")
        cap_effect_excited = cap_effect_laugh = cap_effect_thumbsup = None

    return cap, stream, audio_system, holistic, cap_bg, cap_effect_excited, cap_effect_laugh, cap_effect_thumbsup, frame_h, frame_w

# ============================================
# FUNGSI INI DIUBAH
# ============================================
def initialize_state(frame_w, frame_h):
    """Inisialisasi state termasuk effect state dan UNIFIED IDLE STATE."""
    state = {
        "frame_counter": 0,
        "stable_gesture": "NORMAL",
        "user_is_idle": True,
        "laugh_audio_counter": 0,
        "background_mode": 0,
        "resized_virtual_bg": None,
        "random_blink_timer": random.randint(cfg.RANDOM_BLINK_INTERVAL_MIN_FRAMES, cfg.RANDOM_BLINK_INTERVAL_MAX_FRAMES),
        "random_blink_counter": 0,
        "talk_frame_counter": 0,
        "gesture_buffer": deque(maxlen=cfg.GESTURE_BUFFER_SIZE),
        "current_anchor_pos": (frame_w // 2, frame_h // 2),
        "current_hair_pos": (frame_w // 2, frame_h // 2),
        "hair_velocity": (0, 0),
        "bounce_offset": (0, 0),
        "bounce_velocity": (0, 0),
        
        # --- UNIFIED IDLE STATE ---
        "idle_timer": random.randint(cfg.INITIAL_IDLE_DELAY_MIN_FRAMES, cfg.INITIAL_IDLE_DELAY_MAX_FRAMES),
        "idle_sequence_index": -1, # Index untuk sequence yg aktif
        "idle_mode": "EYES",       # Mode idle: "EYES"
        "current_idle_offset_for_sway": (0, 0),
        "current_eye_direction": "CENTER", # State untuk mata
        "current_eyes": None,      # Menyimpan asset mata yang sedang digunakan
        # --------------------------

        "frame_count_for_process": 0,
        "last_mediapipe_results": None,
        "effect_excited_active": False,
        "effect_laugh_active": False,
        "effect_thumbsup_active": False,
        "last_gesture_for_effect": "NORMAL",
        "current_mouth_state": "closed",
        "target_mouth_state": "closed",
        "mouth_transition_progress": 0.0,
        "silence_frame_counter": 0,
        "last_rms_value": 0,
        "current_rms": 0,
        "current_audio_level": "DIAM",
        "visual_laugh_duration_counter": 0,
        "laugh_cooldown_counter": 0,
        "face_direction_gesture": "NORMAL", # State tambahan untuk deteksi arah wajah
    }
    print("State OK.")
    return state

# --- FUNGSI UTAMA ---

def main():
    assets = None
    cap = stream = audio_system = holistic = cap_bg = None
    cap_effect_excited = cap_effect_laugh = cap_effect_thumbsup = None
    
    try:
        # Panggil fungsi load_assets dari vtuber_core (tanpa argumen)
        assets = load_assets()
        
        cap, stream, audio_system, holistic, cap_bg, cap_effect_excited, cap_effect_laugh, cap_effect_thumbsup, frame_h, frame_w = initialize_systems()
        
        state = initialize_state(frame_w, frame_w)

        print("\n" + "="*60)
        print("🔍 DEBUG MODE: MOUTH DETECTION ACTIVE")
        print("="*60)
        print("Lihat console untuk serial monitor output!")
        print("="*60 + "\n")
        
        # Variabel ini tidak lagi digunakan untuk output, tetapi disimpan sebagai placeholder
        raw_frame_for_mediapipe = None 

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
                
                # Simpan frame asli (meskipun tidak ditampilkan, deteksi MP di bawah memerlukannya)
                raw_frame_for_mediapipe = frame.copy() 
                
                frame = cv2.flip(frame, 1) # Frame yang di-flip untuk avatar
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
            
            # ... (Audio processing tidak berubah) ...
            is_talking_audio, is_audio_laugh_detected, stream = process_audio(state, stream)

            # ... (MediaPipe processing tidak berubah) ...
            results = None
            state["frame_count_for_process"] += 1
            if state["frame_count_for_process"] >= cfg.FRAME_PROCESS_INTERVAL:
                state["frame_count_for_process"] = 0
                if frame is not None and frame.size > 0:
                    # Menggunakan frame yang sudah di-flip untuk deteksi MP (agar landmark sinkron dengan avatar)
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
            
            # ... (Visual lip sync debug tidak berubah) ...
            visual_mouth_state, current_mouth_ratio = calculate_visual_lip_sync(results)

            if state["frame_counter"] % 5 == 0: 
                if results and results.face_landmarks:
                    print(f"📊 [FRAME {state['frame_counter']:04d}] RASIO MULUT: {current_mouth_ratio:.4f} | STATE: {visual_mouth_state} | FACE: ✅ DETECTED")
                else:
                    print(f"⚠️  [FRAME {state['frame_counter']:04d}] FACE: ❌ NOT DETECTED - Pastikan wajah Anda terlihat kamera!")
            
            is_talking = False
            if cfg.LIP_SYNC_MODE == "VISUAL":
                state["target_mouth_state"] = visual_mouth_state
                is_talking = (visual_mouth_state != "closed")
            elif cfg.LIP_SYNC_MODE == "AUDIO":
                is_talking = is_talking_audio
            elif cfg.LIP_SYNC_MODE == "HYBRID":
                state["target_mouth_state"] = visual_mouth_state 
                is_talking = (visual_mouth_state != "closed")
            
            # ... (Logika tawa visual tidak berubah) ...
            is_visual_laugh_detected = False
            if state["laugh_cooldown_counter"] > 0:
                state["laugh_cooldown_counter"] -= 1
            
# ============================================================
# LOGIKA BARU: SUSTAIN LAUGH (TAHAN KETAWA & SHAKING)
# ============================================================
            
            # 1. Cek apakah mulut terbuka lebar (di atas threshold)
            if current_mouth_ratio > cfg.LAUGH_MOUTH_OPENNESS_THRESHOLD:
                # Naikkan counter durasi
                state["visual_laugh_duration_counter"] += 1
                
                # Jika sudah melewati batas minimum durasi (misal 2 frame)
                if state["visual_laugh_duration_counter"] > cfg.LAUGH_DURATION_MIN_FRAMES:
                    is_visual_laugh_detected = True
                    
                    # KUNCI RAHASIA: Jangan reset ke 0!
                    # Kita kunci counter di angka yang aman agar status ketawa AKTIF TERUS
                    state["visual_laugh_duration_counter"] = cfg.LAUGH_DURATION_MIN_FRAMES + 5
                    
                    # Matikan cooldown paksa agar shaking tidak berhenti
                    state["laugh_cooldown_counter"] = 0 
            else:
                # 2. Jika mulut sudah menutup (kurang dari threshold)
                # BARU kita reset counter ke 0.
                # Ini artinya kamu sudah berhenti ketawa.
                state["visual_laugh_duration_counter"] = 0
                is_visual_laugh_detected = False
            
            # ============================================================

            audio_laugh_score = 1.0 if is_audio_laugh_detected else 0.0
            visual_laugh_score = 1.0 if is_visual_laugh_detected else 0.0
            
            combined_laugh_score = (audio_laugh_score * cfg.LAUGH_AUDIO_WEIGHT) + \
                                   (visual_laugh_score * cfg.LAUGH_VISUAL_WEIGHT)
            
            is_laugh_detected = (combined_laugh_score >= cfg.LAUGH_COMBINED_THRESHOLD)

            # ============================================
            # BLOK INI DIUBAH/DITAMBAH (Logika Gestur)
            # ============================================
            is_blinking, force_blink = determine_blink_state(state, results, get_aspect_ratio)
            detect_gestures(state, results)
            
            # --- Deteksi Arah Wajah ---
            face_look_gesture = detect_face_direction(results)
            state["face_direction_gesture"] = face_look_gesture or "NORMAL"
            
            # Panggil SATU FUNGSI IDLE TERPADU
            update_unified_idle_state(state)
            
            # --- Atur Stable Gesture Akhir ---
            current_gesture_from_buffer = state["stable_gesture"]
            
            if current_gesture_from_buffer != "NORMAL":
                pass 
            elif is_laugh_detected:
                state["stable_gesture"] = "LAUGH"
            elif face_look_gesture:
                state["stable_gesture"] = face_look_gesture 
            else:
                state["stable_gesture"] = "NORMAL" 
                
            if is_laugh_detected:
                state["stable_gesture"] = "LAUGH"
            # ============================================
                
            # ... (Effect state management tidak berubah) ...
            current_gesture = state["stable_gesture"]
            last_gesture = state["last_gesture_for_effect"]
            
            if current_gesture != last_gesture:
                state["effect_excited_active"] = (current_gesture == "EXCITED")
                if state["effect_excited_active"]: 
                    if cap_effect_excited: cap_effect_excited.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                state["effect_laugh_active"] = (current_gesture == "LAUGH")
                if state["effect_laugh_active"]:
                    if cap_effect_laugh: cap_effect_laugh.set(cv2.CAP_PROP_POS_FRAMES, 0)

                state["effect_thumbsup_active"] = (current_gesture == "THUMBS_UP")
                if state["effect_thumbsup_active"]:
                    if cap_effect_thumbsup: cap_effect_thumbsup.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                state["last_gesture_for_effect"] = current_gesture
            
            if is_laugh_detected and current_gesture == "LAUGH":
                if not state["effect_laugh_active"]:
                    state["effect_laugh_active"] = True
                    if cap_effect_laugh: cap_effect_laugh.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # ... (Select assets tidak berubah, sudah didesain untuk ini) ...
            current_head_img, current_body_img = select_assets(state, assets, is_blinking, force_blink, is_talking, is_laugh_detected)
            if current_head_img is None or current_body_img is None:
                print("FATAL: Asset selection failed.")
                break

            # ... (Calculate positions tidak berubah) ...
            pos_x_head, pos_y_head, pos_x_body, pos_y_body, pos_x_hair, pos_y_hair = calculate_positions(
                state, results, frame_w, frame_h, current_head_img, current_body_img, assets.get("hair_back")
            )

            # ... (Render tidak berubah) ...
            final_output = draw_background(state, frame_w, frame_h, cap_bg, assets.get("bg_virtual_img"))
            
            if state["effect_excited_active"] and cap_effect_excited and cap_effect_excited.isOpened():
                ret_fx, frame_fx = cap_effect_excited.read()
                if ret_fx and frame_fx is not None:
                    final_output = overlay_video_effect(final_output, frame_fx, 'fullscreen', frame_w, frame_h)
                else:
                    state["effect_excited_active"] = False
            
            if state["effect_laugh_active"] and cap_effect_laugh and cap_effect_laugh.isOpened():
                ret_fx, frame_fx = cap_effect_laugh.read()
                if ret_fx and frame_fx is not None:
                    final_output = overlay_video_effect(final_output, frame_fx, 'fullscreen', frame_w, frame_h)
                else:
                    cap_effect_laugh.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if state["effect_thumbsup_active"] and cap_effect_thumbsup and cap_effect_thumbsup.isOpened():
                ret_fx, frame_fx = cap_effect_thumbsup.read()
                if ret_fx and frame_fx is not None:
                    final_output = overlay_video_effect(final_output, frame_fx, 'fullscreen', frame_w, frame_h)
                else:
                    state["effect_thumbsup_active"] = False
            
            final_output = draw_avatar_and_ui(final_output, state, assets, current_head_img, current_body_img,
                                              pos_x_head, pos_y_head, pos_x_body, pos_y_body,
                                              pos_x_hair, pos_y_hair)
            
            # ============================================
            # BLOK VISUALISASI MEDIAPIPE DIHAPUS/DIABAIKAN
            # cv2.imshow('MediaPipe Detection Output', raw_detection_frame) 
            # ============================================
            
            # ... (Show frame dan cleanup tidak berubah) ...
            try:
                cv2.imshow(cfg.WINDOW_NAME, final_output)
            except Exception as e:
                print(f"Error showing frame: {e}")
                break

            key = cv2.waitKey(5) & 0xFF
            if key in cfg.EXIT_KEYS:
                print("Exit key.")
                break
            elif key == cfg.BG_CHANGE_KEY:
                state["background_mode"] = (state["background_mode"] + 1) % len(cfg.BG_MODES)
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
            "Effect THUMBSUP": cap_effect_thumbsup,
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