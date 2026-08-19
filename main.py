import time

import cv2
import mediapipe as mp

import pyautogui

import math


CURSOR_ENABLED = False

F_MIN = 1
BETA = 4
D_CUTOFF = 0.5

PINCH_THRESHOLD = 0.04
RELEASE_THRESHOLD = 0.05

right_mic_hold_start = None
MIC_HOLD_TIME = 0.2

right_send_hold_start = None
SEND_HOLD_TIME = 0.2

X_MIN = 0.30
X_MAX = 0.70
Y_MIN = 0.30
Y_MAX = 0.70

SWIPE_DISTANCE = 0.08
SWIPE_MAX_TIME = 0.2
SWIPE_MIN_SPEED = 1.2
SWIPE_COOLDOWN = 0.8

SCROLL_INITIAL_SPEED = 3000.0
SCROLL_DECAY = 3.5


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

recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

start_time = time.monotonic()


previous_raw_x = None
previous_raw_y = None

smoothed_x = None
smoothed_y = None

previous_time = None

smoothed_velocity_x = 0.0
smoothed_velocity_y = 0.0

is_pinching = False

right_cursor_hold_start = None
right_cursor_latched = False
right_cursor_release_frames = 0

CURSOR_HOLD_TIME = 0.5
CURSOR_RELEASE_FRAMES = 5

right_mic_latched = False
right_mic_release_frames = 0

right_send_latched = False
right_send_release_frames = 0

COMMAND_RELEASE_FRAMES = 5

swipe_start_y = None
swipe_start_time = None
swipe_block_until = 0.0

scroll_velocity = 0.0
scroll_remainder = 0.0

previous_loop_time = time.monotonic()

last_finger_y = None
last_finger_time = None

while True:
    success, frame = camera.read()

    if not success:
        break

    frame_time = time.monotonic()

    loop_delta_t = frame_time - previous_loop_time
    previous_loop_time = frame_time

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    timestamp_ms = int((time.monotonic() - start_time) * 1000)

    result = recognizer.recognize_for_video(mp_image, timestamp_ms)

    height, width, _ = frame.shape

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[i][0].category_name

        gesture = result.gestures[i][0].category_name

        points = []

        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # LEFT HAND: scrolling only
        if handedness == "Left":
            current_time = frame_time

            finger_y = (
                hand_landmarks[8].y,   # index tip
                hand_landmarks[12].y,  # middle tip
                hand_landmarks[16].y,  # ring tip
            )

            current_time = frame_time

            if current_time >= swipe_block_until:

                if last_finger_y is not None:
                    elapsed = current_time - last_finger_time

                    movements = [
                        current_y - old_y
                        for current_y, old_y
                        in zip(finger_y, last_finger_y)
                    ]

                    if 1e-6 < elapsed < SWIPE_MAX_TIME:

                        speeds = [
                            move / elapsed
                            for move in movements
                        ]

                        if (
                                all(move < -SWIPE_DISTANCE for move in movements)
                                and all(speed < -SWIPE_MIN_SPEED for speed in speeds)
                        ):
                            print("FLICK UP")

                            scroll_velocity = -SCROLL_INITIAL_SPEED
                            swipe_block_until = current_time + SWIPE_COOLDOWN

                        elif (
                                all(move > SWIPE_DISTANCE for move in movements)
                                and all(speed > SWIPE_MIN_SPEED for speed in speeds)
                        ):
                            print("FLICK DOWN")

                            scroll_velocity = SCROLL_INITIAL_SPEED
                            swipe_block_until = current_time + SWIPE_COOLDOWN

            last_finger_y = finger_y
            last_finger_time = current_time

        # RIGHT HAND: cursor + pinch only
        if handedness == "Right":
            # Hold Thumb Down -> toggle cursor
            if gesture == "Thumb_Down":
                right_cursor_release_frames = 0

                if right_cursor_hold_start is None:
                    right_cursor_hold_start = frame_time

                elif (
                        not right_cursor_latched
                        and frame_time - right_cursor_hold_start >= CURSOR_HOLD_TIME
                ):
                    CURSOR_ENABLED = not CURSOR_ENABLED
                    right_cursor_latched = True

                    if CURSOR_ENABLED:
                        pyautogui.moveTo(
                            screen_width // 2,
                            screen_height // 2
                        )

                    print(
                        "CURSOR ON"
                        if CURSOR_ENABLED
                        else "CURSOR OFF"
                    )

            else:
                right_cursor_hold_start = None

                if right_cursor_latched:
                    right_cursor_release_frames += 1

                    if right_cursor_release_frames >= CURSOR_RELEASE_FRAMES:
                        right_cursor_latched = False
                        right_cursor_release_frames = 0
            if not CURSOR_ENABLED:
                # Thumb Up -> toggle mic
                if gesture == "Thumb_Up":
                    right_mic_release_frames = 0

                    if right_mic_hold_start is None:
                        right_mic_hold_start = frame_time

                    elif (
                            not right_mic_latched
                            and frame_time - right_mic_hold_start >= MIC_HOLD_TIME
                    ):
                        pyautogui.hotkey("ctrl", "shift", "d")
                        right_mic_latched = True
                        print("MIC TOGGLE")

                else:
                    right_mic_hold_start = None

                    if right_mic_latched:
                        right_mic_release_frames += 1

                        if right_mic_release_frames >= COMMAND_RELEASE_FRAMES:
                            right_mic_latched = False
                            right_mic_release_frames = 0

                # Hold Open Palm -> press enter
                if gesture == "Open_Palm":
                    right_send_release_frames = 0

                    if right_send_hold_start is None:
                        right_send_hold_start = frame_time

                    elif (
                            not right_send_latched
                            and frame_time - right_send_hold_start >= SEND_HOLD_TIME
                    ):
                        right_send_latched = True
                        pyautogui.press("enter")
                        print("SEND")

                else:
                    right_send_hold_start = None

                    if right_send_latched:
                        right_send_release_frames += 1

                        if right_send_release_frames >= COMMAND_RELEASE_FRAMES:
                            right_send_latched = False
                            right_send_release_frames = 0
            else:
                # Reset command gesture state while cursor mode is ON
                right_mic_latched = False
                right_mic_release_frames = 0
                right_mic_hold_start = None

                right_send_latched = False
                right_send_release_frames = 0
                right_send_hold_start = None

            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]

            pinch_distance = math.sqrt(
                (index_tip.x - thumb_tip.x) ** 2
                + (index_tip.y - thumb_tip.y) ** 2
                + (index_tip.z - thumb_tip.z) ** 2
            )

            if CURSOR_ENABLED:
                if not is_pinching and pinch_distance < PINCH_THRESHOLD:
                    pyautogui.mouseDown()
                    is_pinching = True
                    print("Pinching!")

                elif is_pinching and pinch_distance > RELEASE_THRESHOLD:
                    pyautogui.mouseUp()
                    is_pinching = False
            else:
                # If cursor mode is turned off while dragging,
                # release the mouse immediately.
                if is_pinching:
                    pyautogui.mouseUp()
                    is_pinching = False

            current_time = frame_time

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

            if (
                    CURSOR_ENABLED
                    and gesture not in ("Thumbs_Down", "Thumb_Up", "Open_Palm")
            ):
                pyautogui.moveTo(cursor_x, cursor_y)

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

    # Inertial scrolling runs independently of hand tracking.
    # Even if MediaPipe loses the hand after the flick,
    # the launched scroll keeps moving and decaying.

    if (loop_delta_t >
            0):
        scroll_remainder += scroll_velocity * loop_delta_t

        scroll_steps = int(scroll_remainder)

        if scroll_steps != 0:
            pyautogui.scroll(scroll_steps)
            scroll_remainder -= scroll_steps

        scroll_velocity *= math.exp(-SCROLL_DECAY * loop_delta_t)

        if abs(scroll_velocity) < 1.0:
            scroll_velocity = 0.0
            scroll_remainder = 0.0

    frame = cv2.flip(frame, 1)
    cv2.imshow("Gesture Control", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

recognizer.close()
camera.release()
cv2.destroyAllWindows()
