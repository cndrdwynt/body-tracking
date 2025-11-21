# utils.py
# Berisi fungsi-fungsi utilitas/pembantu.

import cv2
import numpy as np
import math
import mediapipe as mp

# Kita import config untuk mengakses konstanta
import config as cfg

mp_hands = mp.solutions.hands

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

def remove_greenscreen(frame):
    """Hapus background hijau dari video dan kembalikan frame BGRA dengan alpha channel."""
    if frame is None or frame.size == 0:
        return None
    
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Gunakan konstanta dari config
        mask = cv2.inRange(hsv, cfg.CHROMA_KEY_COLOR_LOWER, cfg.CHROMA_KEY_COLOR_UPPER)
        mask_inv = cv2.bitwise_not(mask)
        mask_inv = cv2.GaussianBlur(mask_inv, (5, 5), 0)
        b, g, r = cv2.split(frame)
        frame_bgra = cv2.merge([b, g, r, mask_inv])
        return frame_bgra
    except Exception as e:
        print(f"Greenscreen removal error: {e}")
        return None

def overlay_video_effect(background, video_frame, position='fullscreen', frame_w=None, frame_h=None):
    """Overlay video effect dengan chroma key ke background."""
    if background is None or video_frame is None:
        return background
    
    bg_h, bg_w = background.shape[:2]
    
    effect_with_alpha = remove_greenscreen(video_frame)
    if effect_with_alpha is None:
        return background
    
    if position == 'fullscreen':
        effect_resized = cv2.resize(effect_with_alpha, (bg_w, bg_h))
        return overlay_png(background, effect_resized, 0, 0)
    
    elif position == 'right':
        effect_h, effect_w = effect_with_alpha.shape[:2]
        target_w = int(bg_w * 0.5)
        target_h = int(effect_h * (target_w / effect_w))
        
        if target_h > bg_h:
            target_h = bg_h
            target_w = int(effect_w * (target_h / effect_h))
        
        effect_resized = cv2.resize(effect_with_alpha, (target_w, target_h))
        
        x_pos = bg_w - target_w - 20
        y_pos = (bg_h - target_h) // 2
        
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
            return (lm[idx].x, lm[idx].y) if idx < len(lm) else None
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