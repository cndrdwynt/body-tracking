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
from utils import get_finger_status

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

def load_assets():
    """Memuat dan menskalakan semua aset gambar dari folder yang sesuai."""
    assets = {}
    asset_files = cfg.ASSET_FILES
    
    print("Memuat aset karakter...")
    all_loaded = True
    for name, filename in asset_files.items():
        # Path sekarang menggunakan IMAGE_FOLDER dari config
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

    # --- MEMUAT BACKGROUND IMAGE SECARA TERPISAH ---
    print("Memuat aset background...")
    try:
        # Path sekarang menggunakan BACKGROUND_FOLDER dari config
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
    # --- AKHIR BLOK ---

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

def update_idle_state(state):
    """Update state idle."""
    if state["user_is_idle"]:
        state["idle_timer"] -= 1
        if state["idle_timer"] <= 0:
            state["idle_sequence_index"] = (state["idle_sequence_index"] + 1) % len(cfg.IDLE_SEQUENCE)
            next_s = cfg.IDLE_SEQUENCE[state["idle_sequence_index"]]
            
            if state["stable_gesture"] in ["NORMAL", "LOOK_LEFT_IDLE", "LOOK_RIGHT_IDLE", "LOOK_DOWN_IDLE"]:
                state["stable_gesture"] = next_s
            
            min_d, max_d = cfg.IDLE_STATE_DURATION_RANGES.get(next_s, (cfg.IDLE_LOOK_DURATION_MIN_FRAMES, cfg.IDLE_LOOK_DURATION_MAX_FRAMES))
            state["idle_timer"] = random.randint(min_d, max_d)
    else:
        state["idle_timer"] = random.randint(cfg.INITIAL_IDLE_DELAY_MIN_FRAMES, cfg.INITIAL_IDLE_DELAY_MAX_FRAMES)
        state["idle_sequence_index"] = -1
        state["current_idle_offset_for_sway"] = (0, 0)
        return
    
    target_o = (0, 0)
    cs = state["stable_gesture"]
    if cs == "LOOK_LEFT_IDLE":
        target_o = (-cfg.IDLE_SWAY_TARGET_OFFSET, 0)
    elif cs == "LOOK_RIGHT_IDLE":
        target_o = (cfg.IDLE_SWAY_TARGET_OFFSET, 0)
    elif cs == "LOOK_DOWN_IDLE":
        target_o = (0, cfg.IDLE_SWAY_TARGET_OFFSET)
    
    cx, cy = state["current_idle_offset_for_sway"]
    tx, ty = target_o
    nx = int(cx * (1.0 - cfg.IDLE_SWAY_SPRING_FACTOR) + tx * cfg.IDLE_SWAY_SPRING_FACTOR)
    ny = int(cy * (1.0 - cfg.IDLE_SWAY_SPRING_FACTOR) + ty * cfg.IDLE_SWAY_SPRING_FACTOR)
    
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
                    state["user_is_idle"] = False
                    
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
                    state["user_is_idle"] = False
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
                if lw.visibility > 0.5 and rw.visibility > 0.5 and n.visibility > 0.5 and lw.y < n.y and rw.y < n.y and abs(lw.x - rw.x) < cfg.GESTURE_BLUSH_WRIST_DISTANCE_THRESHOLD:
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
                        
                        if distance < cfg.GESTURE_OK_DISTANCE_THRESHOLD:
                            gesture = "OK"
                        
                        elif finger_status['THUMB'] and not finger_status['INDEX'] and not finger_status['MIDDLE']:
                            gesture = "THUMBS_UP"
                        
                        elif finger_status['INDEX'] and finger_status['MIDDLE'] and not finger_status['RING']:
                            gesture = "PEACE"
                            
                    except (IndexError, AttributeError) as e:
                        print(f"  [ERROR] Error saat cek gesture jari: {e}")

        except IndexError: pass
        except Exception as e:
            print(f"Error saat deteksi gesture (umum): {e}")
    else:
        state["user_is_idle"] = False

    state["gesture_buffer"].append(gesture)
    final_gesture = "NORMAL"
    
    if len(state["gesture_buffer"]) == cfg.GESTURE_BUFFER_SIZE:
        mcg = max(set(state["gesture_buffer"]), key=state["gesture_buffer"].count)
        count = state["gesture_buffer"].count(mcg)
        
        if count >= cfg.GESTURE_CONFIRMATION_THRESHOLD and mcg != "NORMAL":
            final_gesture = mcg
            state["user_is_idle"] = False
        elif state["gesture_buffer"].count("NORMAL") < (cfg.GESTURE_BUFFER_SIZE - cfg.GESTURE_CONFIRMATION_THRESHOLD + 1):
            state["user_is_idle"] = False
    else:
        state["user_is_idle"] = False

    if not state["user_is_idle"]:
        cs = state["stable_gesture"]
        is_idle = cs in ["NORMAL", "LOOK_LEFT_IDLE", "LOOK_RIGHT_IDLE", "LOOK_DOWN_IDLE"]
        new_g = final_gesture if final_gesture != "NORMAL" else "NORMAL"
        if new_g != cs or not is_idle:
            state["stable_gesture"] = new_g
            state["idle_timer"] = random.randint(cfg.INITIAL_IDLE_DELAY_MIN_FRAMES, cfg.INITIAL_IDLE_DELAY_MAX_FRAMES)
            state["idle_sequence_index"] = -1

def determine_blink_state(state, results, get_aspect_ratio_func):
    """Deteksi kedip."""
    is_blinking = False
    if results and results.face_landmarks:
        face_lm = results.face_landmarks
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
    """Pilih aset kepala & badan dengan advanced lip sync."""
    sg = state["stable_gesture"]
    if sg not in assets['gesture_config']:
        sg = "NORMAL"
        state["stable_gesture"] = "NORMAL"
    if is_laugh_detected:
        sg = "LAUGH"
        state["random_blink_counter"] = 0
        state["talk_frame_counter"] = 0

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

    if is_laugh_detected:
        final_head = assets.get("head_laugh", base_head)
    elif sg == "EXCITED":
        final_head = assets.get("head_excited", base_head)
    elif (is_blinking or force_blink):
        blink = variations.get("blink") if variations else None
        if blink is not None:
            final_head = blink
        else:
            fallback_blink = assets.get("head_blush_blink") if sg == "BLUSHING" and assets.get("head_blush_blink") is not None else assets.get("head_normal_blink")
            if fallback_blink is not None:
                final_head = fallback_blink
    elif is_talking and sg == "NORMAL" and variations:
        target_state = state.get("target_mouth_state", "closed")
        mouth_img = variations.get(target_state)
        if mouth_img is not None:
            final_head = mouth_img
            state["current_mouth_state"] = target_state
        else:
            closed_img = variations.get("closed")
            if closed_img is not None:
                final_head = closed_img
            else:
                final_head = base_head
    elif not is_talking and sg == "NORMAL" and variations:
        closed_img = variations.get("closed")
        if closed_img is not None:
            final_head = closed_img
            state["current_mouth_state"] = "closed"

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