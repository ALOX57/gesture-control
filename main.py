import time

import cv2
import mediapipe as mp

import pyautogui

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17),                                   # Palm
]

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
)

screen_width, screen_height = pyautogui.size()
pyautogui.PAUSE = 0

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

start_time = time.monotonic()

while True:
    success, frame = camera.read()

    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    timestamp_ms = int((time.monotonic() - start_time) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    height, width, _ = frame.shape

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[i][0].category_name

        points = []

        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        if handedness == "Right":
            index_tip = hand_landmarks[8]

            cursor_x = int((1 - index_tip.x) * screen_width)
            cursor_y = int(index_tip.y * screen_height)

            cursor_x = max(0, min(screen_width - 1, cursor_x))
            cursor_y = max(0, min(screen_height - 1, cursor_y))

            pyautogui.moveTo(cursor_x, cursor_y)

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    frame = cv2.flip(frame, 1)
    cv2.imshow("Gesture Control", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

landmarker.close()
camera.release()
cv2.destroyAllWindows()
