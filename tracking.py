import cv2
import mediapipe as mp

# inisialisasi mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# buka kamera
cap = cv2.VideoCapture(0)

# inisialisasi holistic
with mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert ke RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        # kembali ke BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # gambar pose (tubuh)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
            )

        # gambar tangan
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # gambar wajah
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1)
            )

        cv2.imshow('Holistic Body + Hands + Face', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # tekan q atau ESC untuk keluar
            break

cap.release()
cv2.destroyAllWindows()
