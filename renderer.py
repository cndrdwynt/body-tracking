# renderer.py
# Berisi fungsi-fungsi untuk menggambar (rendering).

import cv2
import numpy as np

# Import file konfigurasi kita
import config as cfg
# Import utils untuk fungsi pembantu
from utils import overlay_png, draw_fancy_text

def draw_background(state, frame_w, frame_h, cap_bg, bg_virtual_img):
    """Gambar background."""
    bg_mode = state["background_mode"]
    if frame_h <= 0 or frame_w <= 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    fallback = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    fallback[:] = cfg.BG_FALLBACK_COLOR
    
    if bg_mode == 0: # VIRTUAL IMAGE
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
            
    elif bg_mode == 1: # VIRTUAL VIDEO
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
            
    elif bg_mode == 2: # GREEN SCREEN
        bg = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        bg[:] = cfg.BG_GREEN_SCREEN_COLOR
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
    mode_txt = cfg.BG_MODES[bg_idx] if 0 <= bg_idx < len(cfg.BG_MODES) else "BG:?"
    final_output = draw_fancy_text(final_output, mode_txt, cfg.UI_MODE_POS, cfg.UI_FONT, cfg.UI_MODE_SCALE, cfg.UI_TEXT_COLOR, cfg.UI_MODE_THICKNESS, cfg.UI_BG_COLOR, cfg.UI_BG_ALPHA, cfg.UI_PADDING)
    
    exit_keys = [f"'{chr(k)}'" if k < 256 else '?' for k in cfg.EXIT_KEYS if k != 27] + (["'Esc'"] if 27 in cfg.EXIT_KEYS else [])
    exit_txt = "/".join(exit_keys) if exit_keys else "??"
    help_txt = f"'{chr(cfg.BG_CHANGE_KEY)}': BG | {exit_txt}: Keluar"
    final_output = draw_fancy_text(final_output, help_txt, cfg.UI_HELP_POS, cfg.UI_FONT, cfg.UI_HELP_SCALE, cfg.UI_TEXT_COLOR, cfg.UI_HELP_THICKNESS, cfg.UI_BG_COLOR, cfg.UI_BG_ALPHA, cfg.UI_PADDING)
    
    # RMS Monitor
    if cfg.UI_RMS_MONITOR_ENABLED:
        rms_val = state.get("current_rms", 0)
        audio_level = state.get("current_audio_level", "DIAM")
        laugh_counter = state.get("laugh_audio_counter", 0)
        
        if audio_level == "KETAWA!": color = (0, 0, 255)
        elif audio_level == "SANGAT KERAS": color = (0, 165, 255)
        elif audio_level == "KERAS": color = (0, 255, 255)
        elif audio_level == "NORMAL": color = (0, 255, 0)
        elif audio_level == "PELAN": color = (255, 255, 0)
        else: color = (255, 255, 255)
        
        monitor_txt = f"RMS: {rms_val:3d} | Level: {audio_level:12s} | Laugh: {laugh_counter:2d}/{cfg.LAUGH_COUNTER_THRESHOLD}"
        final_output = draw_fancy_text(final_output, monitor_txt, cfg.UI_RMS_POS, cfg.UI_FONT, cfg.UI_RMS_SCALE, color, cfg.UI_RMS_THICKNESS, cfg.UI_BG_COLOR, cfg.UI_BG_ALPHA, cfg.UI_PADDING)
    
    return final_output