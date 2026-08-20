import time

import cv2
import mediapipe as mp
import pyautogui

from config import CURSOR_ENABLED_DEFAULT
from cursor import CursorController
from scrolling import ScrollController
from gestures import GestureController


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


cursor_enabled = CURSOR_ENABLED_DEFAULT


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


options = mp.tasks.vision.GestureRecognizerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="models/gesture_recognizer.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
)


screen_width, screen_height = pyautogui.size()

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


recognizer = (
    mp.tasks.vision.GestureRecognizer.create_from_options(
        options
    )
)


cursor_controller = CursorController()
scroll_controller = ScrollController()
gesture_controller = GestureController()


start_time = time.monotonic()
previous_loop_time = time.monotonic()


while True:
    success, frame = camera.read()

    if not success:
        break

    frame_time = time.monotonic()

    loop_delta_t = (
        frame_time - previous_loop_time
    )

    previous_loop_time = frame_time

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    timestamp_ms = int(
        (time.monotonic() - start_time) * 1000
    )

    result = recognizer.recognize_for_video(
        mp_image,
        timestamp_ms,
    )

    height, width, _ = frame.shape

    for i, hand_landmarks in enumerate(
        result.hand_landmarks
    ):
        handedness = (
            result.handedness[i][0].category_name
        )

        gesture = (
            result.gestures[i][0].category_name
        )

        gesture_controller.process_mic(
            handedness,
            gesture,
            frame_time,

            cursor_enabled,
        )

        gesture_controller.process_send(
            handedness,
            gesture,
            frame_time,
            cursor_enabled,
        )

        points = []

        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1,
            )

        # LEFT HAND
        if handedness == "Left":
            scroll_controller.process_hand(
                hand_landmarks,
                frame_time,
            )

        # RIGHT HAND
        if handedness == "Right":

            cursor_enabled = (
                gesture_controller.process(
                    gesture,
                    frame_time,
                    cursor_enabled,
                    screen_width,
                    screen_height,
                )
            )

            cursor_controller.process(
                hand_landmarks,
                frame_time,
                cursor_enabled,
                gesture,
                screen_width,
                screen_height,
            )

        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2,
            )

    # Inertia remains independent of hand tracking.
    scroll_controller.update_inertia(
        loop_delta_t
    )

    frame = cv2.flip(frame, 1)

    cv2.imshow(
        "Gesture Control",
        frame,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


recognizer.close()
camera.release()
cv2.destroyAllWindows()