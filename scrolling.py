import math

import pyautogui

from config import (
    SWIPE_DISTANCE,
    SWIPE_MAX_TIME,
    SWIPE_MIN_SPEED,
    SWIPE_COOLDOWN,
    SCROLL_INITIAL_SPEED,
    SCROLL_DECAY,
)


class ScrollController:
    def __init__(self):
        self.swipe_block_until = 0.0

        self.last_finger_y = None
        self.last_finger_time = None

        self.scroll_velocity = 0.0
        self.scroll_remainder = 0.0

    def process_hand(self, hand_landmarks, frame_time):
        finger_y = (
            hand_landmarks[8].y,
            hand_landmarks[12].y,
            hand_landmarks[16].y,
        )

        if frame_time >= self.swipe_block_until:

            if self.last_finger_y is not None:
                elapsed = frame_time - self.last_finger_time

                movements = [
                    current_y - old_y
                    for current_y, old_y
                    in zip(finger_y, self.last_finger_y)
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

                        # Reversed direction, as currently desired.
                        self.scroll_velocity = -SCROLL_INITIAL_SPEED
                        self.swipe_block_until = (
                            frame_time + SWIPE_COOLDOWN
                        )

                    elif (
                        all(move > SWIPE_DISTANCE for move in movements)
                        and all(speed > SWIPE_MIN_SPEED for speed in speeds)
                    ):
                        print("FLICK DOWN")

                        # Reversed direction, as currently desired.
                        self.scroll_velocity = SCROLL_INITIAL_SPEED
                        self.swipe_block_until = (
                            frame_time + SWIPE_COOLDOWN
                        )

        self.last_finger_y = finger_y
        self.last_finger_time = frame_time

    def update_inertia(self, delta_t):
        if delta_t <= 0:
            return

        self.scroll_remainder += self.scroll_velocity * delta_t

        scroll_steps = int(self.scroll_remainder)

        if scroll_steps != 0:
            pyautogui.scroll(scroll_steps)
            self.scroll_remainder -= scroll_steps

        self.scroll_velocity *= math.exp(
            -SCROLL_DECAY * delta_t
        )

        if abs(self.scroll_velocity) < 1.0:
            self.scroll_velocity = 0.0
            self.scroll_remainder = 0.0