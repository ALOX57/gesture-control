import time

import cv2
import mediapipe as mp

import pyautogui

import math

F_MIN = 1
BETA = 4
D_CUTOFF = 0.5

X_MIN = 0.30
X_MAX = 0.70
Y_MIN = 0.30
Y_MAX = 0.70


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
pyautogui.FAILSAFE = False

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

start_time = time.monotonic()


previous_raw_x = None
previous_raw_y = None

smoothed_x = None
smoothed_y = None

previous_time = None

smoothed_velocity_x = 0.0
smoothed_velocity_y = 0.0

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

            raw_x = index_tip.x
            raw_y = index_tip.y

            current_time = time.monotonic()

            if previous_time is None:
                previous_raw_x = raw_x
                previous_raw_y = raw_y

                smoothed_x = raw_x
                smoothed_y = raw_y

                previous_time = current_time
            else:
                delta_t = current_time - previous_time

                if delta_t > 1e-6:
                    velocity_x = (raw_x - previous_raw_x) / delta_t
                    velocity_y = (raw_y - previous_raw_y) / delta_t

                    alpha_d = 1 - math.exp(-2 * math.pi * D_CUTOFF * delta_t)

                    smoothed_velocity_x = (
                            alpha_d * velocity_x
                            + (1 - alpha_d) * smoothed_velocity_x
                    )

                    smoothed_velocity_y = (
                            alpha_d * velocity_y
                            + (1 - alpha_d) * smoothed_velocity_y
                    )

                    cutoff_x = F_MIN + BETA * abs(smoothed_velocity_x)
                    cutoff_y = F_MIN + BETA * abs(smoothed_velocity_y)

                    alpha_x = 1 - math.exp(-2 * math.pi * cutoff_x * delta_t)
                    alpha_y = 1 - math.exp(-2 * math.pi * cutoff_y * delta_t)

                    smoothed_x = alpha_x * raw_x + (1 - alpha_x) * smoothed_x
                    smoothed_y = alpha_y * raw_y + (1 - alpha_y) * smoothed_y

                previous_raw_x = raw_x
                previous_raw_y = raw_y
                previous_time = current_time

            mapped_x = (smoothed_x - X_MIN) / (X_MAX - X_MIN)
            mapped_x = max(0, min(1, mapped_x))

            cursor_x = int((1 - mapped_x) * screen_width)

            mapped_y = (smoothed_y - Y_MIN) / (Y_MAX - Y_MIN)
            mapped_y = max(0, min(1, mapped_y))

            cursor_y = int(mapped_y * screen_height)

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
