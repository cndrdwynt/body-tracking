# vtuber_core.py
# Berisi semua logika inti untuk VTuber.

import cv2
import os
import random
import math
import audioop
import mediapipe as mp
from collections import deque

# Import file konfigurasi kita
import config as cfg
# Import utils untuk fungsi pembantu
from utils import get_finger_status, get_aspect_ratio

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

def load_assets():
    """Memuat dan menskalakan semua aset gambar dari folder yang sesuai."""
    assets = {}
    asset_files = cfg.ASSET_FILES
    
    print("Memuat aset karakter...")
    all_loaded = True
    for name, filename in asset_files.items():
        path = os.path.join(cfg.IMAGE_FOLDER, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            if "mouth_" in name:
                print(f"Warn: '{filename}' tidak ditemukan, gunakan default")
                assets[name] = assets.get("head_normal")
                continue
            print(f"Error load: '{filename}'")
            all_loaded = False
            assets[name] = None
            continue
        
        is_head = "head" in name or "mouth" in name
        is_hair = "hair" in name
        is_body = "body" in name
        
        scale = cfg.HEAD_SCALE_FACTOR if (is_head or is_hair) else cfg.BODY_SCALE_FACTOR if is_body else 1.0
        
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

        if assets.get(name) is not None:
            print(f" - OK: {filename}")
            
    if not all_loaded:
        raise FileNotFoundError("Aset karakter penting hilang/gagal.")

    # ============= ASET MATA OVERLAY =============
    print("Memuat aset mata overlay...")
    for eye_name, eye_file in cfg.EYE_OVERLAY_FILES.items():
        path = os.path.join(cfg.IMAGE_FOLDER, eye_file)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"Warn: '{eye_file}' tidak ditemukan, akan skip eye overlay")
            assets[eye_name] = None
        else:
            try:
                # Scale dengan HEAD_SCALE_FACTOR
                scaled = cv2.resize(img, (0, 0), 
                                    fx=cfg.HEAD_SCALE_FACTOR, 
                                    fy=cfg.HEAD_SCALE_FACTOR, 
                                    interpolation=cv2.INTER_AREA)
                assets[eye_name] = scaled
                print(f" - OK: {eye_file}")
            except Exception as e:
                print(f"Error resize '{eye_file}': {e}")
                assets[eye_name] = None
    # ===============================================

    print("Memuat aset background...")
    try:
        bg_img_path = os.path.join(cfg.BACKGROUND_FOLDER, cfg.BG_VIRTUAL_IMAGE_PATH)
        bg_img = cv2.imread(bg_img_path)
        if bg_img is None:
            print(f"Warn: Gagal memuat background image: '{bg_img_path}'")
            assets['bg_virtual_img'] = None
        else:
            assets['bg_virtual_img'] = bg_img
            print(f" - OK: {cfg.BG_VIRTUAL_IMAGE_PATH}")
    except Exception as e:
        print(f"Error load BG image: {e}")
        assets['bg_virtual_img'] = None

    # Membuat Map Variasi Kepala
    assets['head_variations'] = {}
    for base_name, variation_names in cfg.HEAD_VARIATION_MAP_CONFIG.items():
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
    for gesture_name, head_name, body_name in cfg.GESTURE_CONFIG_SETUP:
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

def update_unified_idle_state(state):
    """
    Update state idle terpadu (menggantikan update_idle_state dan update_eye_idle_state).
    Akan bergantian antara idle "BODY" (tengok kepala) dan "EYES" (lirik pupil).
    CATATAN: Logika idle BODY TENGOK KEPALA TELAH DIHAPUS dan digantikan oleh deteksi arah wajah.
    """
    
    # 1. Handle Reset (jika user tidak idle)
    if not state["user_is_idle"]:
        # Reset timer utama
        state["idle_timer"] = random.randint(cfg.INITIAL_IDLE_DELAY_MIN_FRAMES, cfg.INITIAL_IDLE_DELAY_MAX_FRAMES)
        # Selalu mulai dari mode EYES
        state["idle_mode"] = "EYES" 
        state["idle_sequence_index"] = -1 # Akan di-increment ke 0 saat idle pertama
        # Reset semua state ke default
        state["current_eye_direction"] = "CENTER"
        state["current_idle_offset_for_sway"] = (0, 0)
        return

    # 2. Handle Sway (body sway SANGAT kecil)
    target_o = (0, 0)
    
    cx, cy = state["current_idle_offset_for_sway"]
    tx, ty = target_o
    nx = int(cx * (1.0 - cfg.IDLE_SWAY_SPRING_FACTOR) + tx * cfg.IDLE_SWAY_SPRING_FACTOR)
    ny = int(cy * (1.0 - cfg.IDLE_SWAY_SPRING_FACTOR) + ty * cfg.IDLE_SWAY_SPRING_FACTOR)
    state["current_idle_offset_for_sway"] = (nx, ny)


    # 3. Decrement Timer
    state["idle_timer"] -= 1

    # 4. Check Timer Expiry (jika timer habis, ganti ke aksi idle berikutnya)
    if state["idle_timer"] <= 0:
        current_mode = state.get("idle_mode", "EYES")
        
        # Di sini, kita hanya fokus ke mode EYES. Mode BODY idle sudah dihapus.
        if current_mode == "EYES":
            # Kita ada di mode EYES, jalankan sequence EYES (lirik pupil)
            state["idle_sequence_index"] = (state["idle_sequence_index"] + 1)
            
            if state["idle_sequence_index"] >= len(cfg.EYE_IDLE_SEQUENCE):
                # Sequen EYES selesai, ulangi sequence EYES setelah delay
                # Kita tidak kembali ke mode "BODY" lagi
                state["idle_mode"] = "EYES"
                state["idle_sequence_index"] = 0 # Mulai dari awal sequence EYES
                state["current_eye_direction"] = "CENTER" # Pastikan mata normal
                
                # Atur timer untuk delay sebelum idle eye berikutnya
                state["idle_timer"] = random.randint(cfg.EYE_IDLE_DELAY_MIN_FRAMES, cfg.EYE_IDLE_DELAY_MAX_FRAMES)
            else:
                # Masih di sequence EYES
                next_eye_dir = cfg.EYE_IDLE_SEQUENCE[state["idle_sequence_index"]]
                state["current_eye_direction"] = next_eye_dir
                # Atur timer untuk aksi mata ini
                if next_eye_dir == "CENTER":
                    state["idle_timer"] = cfg.EYE_RETURN_TO_CENTER_FRAMES
                else:
                    state["idle_timer"] = random.randint(cfg.EYE_LOOK_DURATION_MIN_FRAMES, cfg.EYE_LOOK_DURATION_MAX_FRAMES)


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
            if available >= cfg.AUDIO_CHUNK_SIZE:
                audio_data = stream.read(cfg.AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                rms = audioop.rms(audio_data, 2)
                
                smoothed_rms = int(state["last_rms_value"] * (1 - cfg.LIP_SYNC_SMOOTH_FACTOR) + rms * cfg.LIP_SYNC_SMOOTH_FACTOR)
                state["last_rms_value"] = smoothed_rms
                state["current_rms"] = smoothed_rms

                if smoothed_rms < cfg.AUDIO_SPEAK_THRESHOLD_RMS:
                    audio_level = "DIAM"
                    state["silence_frame_counter"] += 1
                    if state["silence_frame_counter"] > cfg.LIP_SYNC_HOLD_FRAMES:
                        state["target_mouth_state"] = "closed"
                else:
                    state["silence_frame_counter"] = 0
                    is_talking = True
                    #state["user_is_idle"] = False
                    
                    if smoothed_rms < cfg.AUDIO_RMS_LEVEL_SMALL:
                        state["target_mouth_state"] = "small"
                        audio_level = "PELAN"
                    elif smoothed_rms < cfg.AUDIO_RMS_LEVEL_MEDIUM:
                        state["target_mouth_state"] = "medium"
                        audio_level = "NORMAL"
                    elif smoothed_rms < cfg.AUDIO_RMS_LEVEL_WIDE:
                        state["target_mouth_state"] = "wide"
                        audio_level = "KERAS"
                    else:
                        if state["frame_counter"] % 10 < 5:
                            state["target_mouth_state"] = "wide"
                        else:
                            state["target_mouth_state"] = "o"
                        audio_level = "SANGAT KERAS"

                if smoothed_rms > cfg.LAUGH_RMS_THRESHOLD:
                    state["laugh_audio_counter"] = min(state["laugh_audio_counter"] + 1, cfg.LAUGH_COUNTER_MAX)
                else:
                    state["laugh_audio_counter"] = max(state["laugh_audio_counter"] - cfg.LAUGH_COUNTER_DECREMENT, 0)

                if state["laugh_audio_counter"] > cfg.LAUGH_COUNTER_THRESHOLD:
                    is_laugh_detected = True
                    is_talking = False
                    #state["user_is_idle"] = False
                    audio_level = "KETAWA!"

                state["current_audio_level"] = audio_level

        except IOError as e:
            if e.errno == -9981: pass
            else:
                print(f"Warn: Audio IO Err: {e}. Disabling.")
                try:
                    if stream.is_active():
                        stream.stop_stream()
                        stream.close()
                except Exception: pass
                stream = None
        except Exception as e:
            print(f"Audio Err: {e}")
            try:
                if stream.is_active():
                    stream.stop_stream()
                    stream.close()
            except Exception: pass
            stream = None

    return is_talking, is_laugh_detected, stream

def detect_face_direction(results):
    """
    Mendeteksi arah tengok kepala berdasarkan posisi hidung (NOSE) relatif terhadap bahu (SHOULDER).
    Mengembalikan: "LOOK_LEFT_IDLE", "LOOK_RIGHT_IDLE", "LOOK_DOWN_IDLE", atau None.
    """
    pose_ok = results and results.pose_landmarks and hasattr(results.pose_landmarks, 'landmark')
    if not pose_ok:
        return None
    
    pose = results.pose_landmarks.landmark
    
    try:
        nose = pose[mp_holistic.PoseLandmark.NOSE]
        left_shoulder = pose[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        
        # Periksa visibilitas
        if nose.visibility < 0.5 or left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
            return None
        
        # Hitung titik tengah bahu
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        
        # Perbedaan horizontal (X) antara hidung dan titik tengah bahu
        diff_x = nose.x - shoulder_mid_x
        
        # Deteksi tengok kiri/kanan
        if diff_x > cfg.FACE_DIRECTION_THRESHOLD:
            # Hidung lebih ke kanan dari bahu -> user menghadap ke kiri (di kamera)
            print("👁️  LOOK_LEFT_IDLE DETECTED (Face Direction)")
            return "LOOK_LEFT_IDLE"
        elif diff_x < -cfg.FACE_DIRECTION_THRESHOLD:
            # Hidung lebih ke kiri dari bahu -> user menghadap ke kanan (di kamera)
            print("👁️  LOOK_RIGHT_IDLE DETECTED (Face Direction)")
            return "LOOK_RIGHT_IDLE"

        # Deteksi tengok bawah (diabaikan dulu, fokus ke kiri/kanan lebih akurat)
        # diff_y = nose.y - left_shoulder.y
        # if diff_y > 0.05:
        #    print("👁️  LOOK_DOWN_IDLE DETECTED (Face Direction)")
        #    return "LOOK_DOWN_IDLE"

    except (IndexError, AttributeError) as e:
        print(f"[ERROR] Face Direction: {e}")
        return None

    return None

def detect_gestures(state, results):
    """
    Deteksi gesture dengan logika STRICT:
    - Wave: 1 Tangan diatas bahu (strict).
    - Blushing: Tangan menyilang.
    - OK: Telunjuk Tekuk + 3 Jari Lain Lurus (SIMPLE LOGIC).
    """
    gesture = "NORMAL"
    
    pose_ok = results and results.pose_landmarks and hasattr(results.pose_landmarks, 'landmark')
    
    if pose_ok:
        pose = results.pose_landmarks.landmark
        left_h = results.left_hand_landmarks
        right_h = results.right_hand_landmarks
        
        # Cek keberadaan tangan
        left_ok = left_h and hasattr(left_h, 'landmark')
        right_ok = right_h and hasattr(right_h, 'landmark')

        # Cek Idle (Hidung)
        try:
            if pose[mp_holistic.PoseLandmark.NOSE].visibility < 0.5:
                state["user_is_idle"] = False
        except IndexError:
            state["user_is_idle"] = False

        try:
            # Ambil Landmark Penting
            rw_lm = pose[mp_holistic.PoseLandmark.RIGHT_WRIST]
            rs_lm = pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            lw_lm = pose[mp_holistic.PoseLandmark.LEFT_WRIST]
            ls_lm = pose[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            nose_lm = pose[mp_holistic.PoseLandmark.NOSE]

            # --- 1. CEK POSISI TANGAN (NAIK / TURUN) ---
            r_is_up = rw_lm.visibility > 0.5 and rs_lm.visibility > 0.5 and rw_lm.y < rs_lm.y
            l_is_up = lw_lm.visibility > 0.5 and ls_lm.visibility > 0.5 and lw_lm.y < ls_lm.y

            # --- 2. DETEKSI BLUSHING (PRIORITAS TERTINGGI) ---
            wrists_crossed = False
            if rw_lm.visibility > 0.5 and lw_lm.visibility > 0.5:
                wrist_dist = abs(rw_lm.x - lw_lm.x)
                near_nose = rw_lm.y < nose_lm.y + 0.1 and lw_lm.y < nose_lm.y + 0.1
                if wrist_dist < 0.15 and near_nose:
                    wrists_crossed = True

            if wrists_crossed:
                gesture = "BLUSHING"
                print("✅ BLUSHING DETECTED! (Tangan Menyilang)")

            # --- 3. DETEKSI WAVE vs EXCITED ---
            if gesture == "NORMAL":
                # Cek Jari Terbuka (Open Palm)
                r_is_open = False
                l_is_open = False
                if right_ok:
                    r_stat = get_finger_status(right_h)
                    r_is_open = r_stat['INDEX'] and r_stat['MIDDLE'] and r_stat['RING'] and r_stat['PINKY']
                if left_ok:
                    l_stat = get_finger_status(left_h)
                    l_is_open = l_stat['INDEX'] and l_stat['MIDDLE'] and l_stat['RING'] and l_stat['PINKY']

                if r_is_up and l_is_up:
                    gesture = "EXCITED"
                    print("✅ EXCITED DETECTED! (Dua Tangan Naik)")
                elif (r_is_up and r_is_open and not l_is_up):
                    gesture = "WAVE"
                    print("👋 WAVE DETECTED! (Hanya Tangan Kanan)")
                elif (l_is_up and l_is_open and not r_is_up):
                    gesture = "WAVE"
                    print("👋 WAVE DETECTED! (Hanya Tangan Kiri)")

            # --- 4. DETEKSI JARI (THUMBS UP, PEACE, OK) ---
            # Hanya dijalankan jika tangan TIDAK diatas bahu (Gesture Normal)
            if gesture == "NORMAL":
                
                # Fungsi helper lokal untuk cek jari
                def check_finger_gesture(hand_landmarks):
                    fs = get_finger_status(hand_landmarks)
                    
                    # A. THUMBS UP: Jempol Lurus, SISANYA TEKUK
                    if fs['THUMB'] and \
                       not fs['INDEX'] and not fs['MIDDLE'] and \
                       not fs['RING'] and not fs['PINKY']:
                        return "THUMBS_UP"

                    # B. PEACE: Telunjuk & Tengah Lurus, Sisanya Tekuk
                    if fs['INDEX'] and fs['MIDDLE'] and \
                       not fs['RING'] and not fs['PINKY']:
                        return "PEACE"

                    # C. OK (VERSI SIMPLE)
                    # Telunjuk DITEKUK, 3 Jari Lain LURUS
                    if not fs['INDEX'] and \
                       fs['MIDDLE'] and fs['RING'] and fs['PINKY']:
                        return "OK"
                        
                    return None

                # Cek Tangan Kanan
                if right_ok:
                    g = check_finger_gesture(right_h)
                    if g:
                        gesture = g
                        print(f"✅ {g} DETECTED! (Right Hand)")
                
                # Cek Tangan Kiri (Fallback)
                elif left_ok:
                    g = check_finger_gesture(left_h)
                    if g:
                        gesture = g
                        print(f"✅ {g} DETECTED! (Left Hand)")
                            
        except (IndexError, AttributeError) as e:
            print(f"[ERROR] Gesture Logic: {e}")
        except Exception as e:
            print(f"Error gesture umum: {e}")
    else:
        state["user_is_idle"] = False

    # --- PROSES BUFFER ---
    state["gesture_buffer"].append(gesture)
    if len(state["gesture_buffer"]) == cfg.GESTURE_BUFFER_SIZE:
        counts = {}
        for g in state["gesture_buffer"]:
            counts[g] = counts.get(g, 0) + 1
        current_stable = state.get("stable_gesture", "NORMAL")
        most_common = max(counts, key=counts.get)
        
        if counts[most_common] >= cfg.GESTURE_CONFIRMATION_THRESHOLD:
             if not most_common.startswith("LOOK_"): 
                state["stable_gesture"] = most_common
        elif current_stable not in ["NORMAL", "LAUGH"] and counts.get("NORMAL", 0) >= cfg.GESTURE_CONFIRMATION_THRESHOLD:
             state["stable_gesture"] = "NORMAL"

    if not state["user_is_idle"]:
        state["stable_gesture"] = gesture

def determine_blink_state(state, results, get_aspect_ratio_func):
    """Deteksi kedip."""
    is_blinking = False
    if results and results.face_landmarks:
        face_lm = results.face_landmarks
        # Menggunakan landmark wajah yang lebih stabil untuk deteksi mata
        le = get_aspect_ratio_func(face_lm, 386, 374, 362, 263)
        re = get_aspect_ratio_func(face_lm, 159, 145, 133, 33)
        if le > 1e-6 and re > 1e-6 and (le + re) / 2.0 < cfg.EYE_AR_THRESHOLD:
            is_blinking = True
            state["random_blink_timer"] = random.randint(cfg.RANDOM_BLINK_INTERVAL_MIN_FRAMES, cfg.RANDOM_BLINK_INTERVAL_MAX_FRAMES)
            state["random_blink_counter"] = 0
            
    force_blink = False
    if state["random_blink_counter"] > 0:
        force_blink = True
        state["random_blink_counter"] -= 1
    else:
        state["random_blink_timer"] -= 1
        
    if state["random_blink_timer"] <= 0:
        force_blink = True
        state["random_blink_counter"] = cfg.RANDOM_BLINK_DURATION_FRAMES
        state["random_blink_timer"] = random.randint(cfg.RANDOM_BLINK_INTERVAL_MIN_FRAMES, cfg.RANDOM_BLINK_INTERVAL_MAX_FRAMES)
        
    return is_blinking, force_blink

def select_assets(state, assets, is_blinking, force_blink, is_talking, is_laugh_detected):
    """Pilih aset kepala & badan dengan SMOOTH lip sync transition + EYE OVERLAY."""
    
    # Ambil gesture stabil saat ini, atau set ke NORMAL jika tidak ada
    sg = state["stable_gesture"]
    if sg not in assets['gesture_config']:
        sg = "NORMAL"
        state["stable_gesture"] = "NORMAL"

    # Force LAUGH jika terdeteksi
    if is_laugh_detected:
        sg = "LAUGH"
        state["random_blink_counter"] = 0
        state["talk_frame_counter"] = 0

    # Ambil konfigurasi aset (head/body)
    cfg_assets = assets['gesture_config'].get(sg, assets['gesture_config']["NORMAL"])
    body_img, base_head = cfg_assets.get("body"), cfg_assets.get("head")
    
    if base_head is None or body_img is None:
        print(f"FATAL: Missing base assets '{sg}'. Using NORMAL.")
        cfg_assets = assets['gesture_config']["NORMAL"]
        body_img, base_head = cfg_assets.get("body"), cfg_assets.get("head")
    if base_head is None or body_img is None:
        print("FATAL: Missing NORMAL assets!")
        return None, None

    final_head = base_head
    variations = assets['head_variations'].get(id(base_head))

    # --- LOGIKA PILIH HEAD ASSET ---
    if is_laugh_detected:
        final_head = assets.get("head_laugh", base_head)
        state["current_mouth_state"] = "laugh"
    elif sg == "EXCITED":
        final_head = assets.get("head_excited", base_head)
        state["current_mouth_state"] = "excited"
    elif (is_blinking or force_blink):
        # Kedip
        blink = variations.get("blink") if variations else None
        if blink is not None:
            final_head = blink
        else:
            fallback_blink = assets.get("head_blush_blink") if sg == "BLUSHING" and assets.get("head_blush_blink") is not None else assets.get("head_normal_blink")
            if fallback_blink is not None:
                final_head = fallback_blink
    elif sg in ["LOOK_LEFT_IDLE", "LOOK_RIGHT_IDLE", "LOOK_DOWN_IDLE"]:
        # Gunakan aset kepala tengok yang terdeteksi
        pass # final_head sudah diset di atas dari cfg_assets
    elif variations:
        # ===================================================================
        # SMOOTH MOUTH TRANSITION SYSTEM (Hanya untuk kepala 'head_normal')
        # ===================================================================
        target_state = state.get("target_mouth_state", "closed")
        current_state = state.get("current_mouth_state", "closed")
        
        # Definisikan urutan state mulut (dari paling kecil ke besar)
        mouth_sequence = ["closed", "small", "medium", "wide", "o"]
        
        try:
            target_idx = mouth_sequence.index(target_state)
            current_idx = mouth_sequence.index(current_state)
        except ValueError:
            target_idx = 0
            current_idx = 0
        
        # Transisi bertahap (naik/turun 1 step per frame)
        if current_idx < target_idx:
            new_idx = current_idx + 1
        elif current_idx > target_idx:
            new_idx = current_idx - 1
        else:
            new_idx = current_idx
        
        # Update current state
        new_state = mouth_sequence[new_idx]
        state["current_mouth_state"] = new_state
        
        # Ambil asset sesuai state baru
        mouth_img = variations.get(new_state)
        if mouth_img is not None:
            final_head = mouth_img
        else:
            closed_img = variations.get("closed")
            final_head = closed_img if closed_img is not None else base_head
        
        # Debug (opsional, bisa dimatikan)
        # if target_state != current_state:
        #     print(f"[SMOOTH MOUTH] {current_state} -> {new_state} (target: {target_state})")


    # ============= LOGIKA EYE OVERLAY =============
    selected_eyes = None
    eye_direction = state.get("current_eye_direction", "CENTER")
    
    if is_blinking or force_blink:
        # Saat blink, TIDAK TAMPILKAN MATA (None)
        selected_eyes = None
    elif is_laugh_detected or sg == "LAUGH":
        # Saat ketawa, mata normal (jika ada overlay)
        selected_eyes = assets.get("eyes_normal")
    elif sg in ["LOOK_LEFT_IDLE", "LOOK_RIGHT_IDLE", "LOOK_DOWN_IDLE"]:
        # Saat gestur tengok/nunduk, JANGAN gunakan eye overlay (karena aset kepala sudah 'tengok')
        selected_eyes = None
    else:
        # JIKA gestur NORMAL/lainnya, baru kita pakai lirikan mata (idle/tengah)
        if eye_direction == "LEFT":
            selected_eyes = assets.get("eyes_look_left")
        elif eye_direction == "RIGHT":
            selected_eyes = assets.get("eyes_look_right")
        else:  # CENTER atau state lainnya
            selected_eyes = assets.get("eyes_normal")
    
    # Fallback jika asset tidak ada
    if selected_eyes is None and not (is_blinking or force_blink or sg in ["LOOK_LEFT_IDLE", "LOOK_RIGHT_IDLE", "LOOK_DOWN_IDLE"]):
         # (Jangan fallback ke mata normal jika memang sengaja di-set None oleh gestur tengok)
        selected_eyes = assets.get("eyes_normal")
    
    # Simpan ke state untuk digunakan saat rendering
    state["current_eyes"] = selected_eyes
    # ======================================================

    if final_head is None: final_head = assets.get("head_normal")
    if body_img is None: body_img = assets.get("body_normal")
        
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
    ax = int(cx * (1.0 - cfg.POSE_SPRING_FACTOR) + tx * cfg.POSE_SPRING_FACTOR)
    ay = int(cy * (1.0 - cfg.POSE_SPRING_FACTOR) + ty * cfg.POSE_SPRING_FACTOR)
    state["current_anchor_pos"] = (ax, ay)

    target_hair_x = ax + cfg.HAIR_OFFSET_X
    target_hair_y = ay + cfg.HAIR_OFFSET_Y
    
    chx, chy = state["current_hair_pos"]
    
    force_x = (target_hair_x - chx) * cfg.HAIR_SPRING_FACTOR
    force_y = (target_hair_y - chy) * cfg.HAIR_SPRING_FACTOR
    
    vel_x, vel_y = state.get("hair_velocity", (0, 0))
    vel_x = (vel_x + force_x) * cfg.HAIR_DRAG_FACTOR
    vel_y = (vel_y + force_y) * cfg.HAIR_DRAG_FACTOR
    
    new_hair_x = chx + vel_x
    new_hair_y = chy + vel_y
    
    dx = new_hair_x - target_hair_x
    dy = new_hair_y - target_hair_y
    distance = math.sqrt(dx * dx + dy * dy)
    if distance > cfg.HAIR_MAX_DISTANCE:
        scale = cfg.HAIR_MAX_DISTANCE / distance
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

    pxh = ax - (hw // 2) + cfg.HEAD_GLOBAL_OFFSET[0]
    pyh = ay - (hh // 2) + cfg.HEAD_GLOBAL_OFFSET[1]
    pxb = ax - (bw // 2) + cfg.BODY_OFFSET_FROM_HEAD_X
    pyb = pyh + cfg.BODY_OFFSET_FROM_HEAD_Y
    pxhr = hax - (hrw // 2)
    pyhr = hay - (hrh // 2)

    fc = state["frame_counter"]
    swy = int(math.sin(fc * cfg.BREATH_SWAY_Y_FREQ) * cfg.BREATH_SWAY_Y_AMP)
    swx = int(math.cos(fc * cfg.BREATH_SWAY_X_FREQ) * cfg.BREATH_SWAY_X_AMP)
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
        shx = int(random.uniform(-cfg.LAUGH_SHAKE_AMOUNT_X, cfg.LAUGH_SHAKE_AMOUNT_X))
        shy = int(random.uniform(-cfg.LAUGH_SHAKE_AMOUNT_Y, cfg.LAUGH_SHAKE_AMOUNT_Y))
        pyh += shy
        pyb += shy
        pyhr += shy
        pxh += shx
        pxb += shx
        pxhr += shx

    return int(pxh), int(pyh), int(pxb), int(pyb), int(pxhr), int(pyhr) 

def calculate_visual_lip_sync(results):
    """
    Menghitung rasio mulut dari landmark visual dan menentukan state-nya.
    DENGAN SMOOTHING untuk mengurangi jitter.
    """
    current_mouth_ratio = 0.0
    visual_mouth_state = "closed"

    if results and results.face_landmarks:
        face_lm = results.face_landmarks
        
        ratio = get_aspect_ratio(face_lm, 
                                 cfg.MOUTH_LANDMARK_UPPER,
                                 cfg.MOUTH_LANDMARK_LOWER,
                                 cfg.MOUTH_LANDMARK_LEFT,
                                 cfg.MOUTH_LANDMARK_RIGHT)
        
        current_mouth_ratio = ratio

        # Tentukan state berdasarkan threshold dengan HYSTERESIS
        # (menambah sedikit "dead zone" untuk mencegah flicker)
        if ratio > cfg.MOUTH_OPENNESS_THRESHOLD_O + 0.05:
            visual_mouth_state = "o"
        elif ratio > cfg.MOUTH_OPENNESS_THRESHOLD_WIDE + 0.03:
            visual_mouth_state = "wide"
        elif ratio > cfg.MOUTH_OPENNESS_THRESHOLD_MEDIUM + 0.02:
            visual_mouth_state = "medium"
        elif ratio > cfg.MOUTH_OPENNESS_THRESHOLD_SMALL + 0.01:
            visual_mouth_state = "small"
        else:
            visual_mouth_state = "closed"
    
    return visual_mouth_state, current_mouth_ratio