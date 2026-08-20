import math

import pyautogui

from config import (
    F_MIN,
    BETA,
    D_CUTOFF,
    PINCH_THRESHOLD,
    RELEASE_THRESHOLD,
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
)


class CursorController:
    def __init__(self):
        self.previous_raw_x = None
        self.previous_raw_y = None

        self.smoothed_x = None
        self.smoothed_y = None

        self.previous_time = None

        self.smoothed_velocity_x = 0.0
        self.smoothed_velocity_y = 0.0

        self.is_pinching = False

    def process(
        self,
        hand_landmarks,
        frame_time,
        cursor_enabled,
        gesture,
        screen_width,
        screen_height,
    ):
        index_tip = hand_landmarks[8]
        thumb_tip = hand_landmarks[4]

        # --------------------------------------------------
        # Pinch
        # --------------------------------------------------

        pinch_distance = math.sqrt(
            (index_tip.x - thumb_tip.x) ** 2
            + (index_tip.y - thumb_tip.y) ** 2
            + (index_tip.z - thumb_tip.z) ** 2
        )

        if cursor_enabled:
            if (
                not self.is_pinching
                and pinch_distance < PINCH_THRESHOLD
            ):
                pyautogui.mouseDown()
                self.is_pinching = True

            elif (
                self.is_pinching
                and pinch_distance > RELEASE_THRESHOLD
            ):
                pyautogui.mouseUp()
                self.is_pinching = False

        else:
            # Never leave the mouse held after cursor mode turns off.
            if self.is_pinching:
                pyautogui.mouseUp()
                self.is_pinching = False

        # --------------------------------------------------
        # Adaptive cursor filter
        # --------------------------------------------------

        raw_x = index_tip.x
        raw_y = index_tip.y

        if self.previous_time is None:
            self.previous_raw_x = raw_x
            self.previous_raw_y = raw_y

            self.smoothed_x = raw_x
            self.smoothed_y = raw_y

            self.previous_time = frame_time

        else:
            delta_t = frame_time - self.previous_time

            if delta_t > 1e-6:
                velocity_x = (
                    raw_x - self.previous_raw_x
                ) / delta_t

                velocity_y = (
                    raw_y - self.previous_raw_y
                ) / delta_t

                alpha_d = 1 - math.exp(
                    -2 * math.pi * D_CUTOFF * delta_t
                )

                self.smoothed_velocity_x = (
                    alpha_d * velocity_x
                    + (1 - alpha_d)
                    * self.smoothed_velocity_x
                )

                self.smoothed_velocity_y = (
                    alpha_d * velocity_y
                    + (1 - alpha_d)
                    * self.smoothed_velocity_y
                )

                cutoff_x = (
                    F_MIN
                    + BETA * abs(self.smoothed_velocity_x)
                )

                cutoff_y = (
                    F_MIN
                    + BETA * abs(self.smoothed_velocity_y)
                )

                alpha_x = 1 - math.exp(
                    -2 * math.pi * cutoff_x * delta_t
                )

                alpha_y = 1 - math.exp(
                    -2 * math.pi * cutoff_y * delta_t
                )

                self.smoothed_x = (
                    alpha_x * raw_x
                    + (1 - alpha_x) * self.smoothed_x
                )

                self.smoothed_y = (
                    alpha_y * raw_y
                    + (1 - alpha_y) * self.smoothed_y
                )

            self.previous_raw_x = raw_x
            self.previous_raw_y = raw_y
            self.previous_time = frame_time

        # --------------------------------------------------
        # Camera -> monitor mapping
        # --------------------------------------------------

        mapped_x = (
            self.smoothed_x - X_MIN
        ) / (X_MAX - X_MIN)

        mapped_x = max(0, min(1, mapped_x))

        cursor_x = int(
            (1 - mapped_x) * screen_width
        )

        mapped_y = (
            self.smoothed_y - Y_MIN
        ) / (Y_MAX - Y_MIN)

        mapped_y = max(0, min(1, mapped_y))

        cursor_y = int(
            mapped_y * screen_height
        )

        cursor_x = max(
            0,
            min(screen_width - 1, cursor_x),
        )

        cursor_y = max(
            0,
            min(screen_height - 1, cursor_y),
        )

        # Don't move cursor while performing command poses.
        if (
            cursor_enabled
            and gesture
            not in (
                "Thumb_Down",
                "Thumb_Up",
                "Open_Palm",
            )
        ):
            pyautogui.moveTo(
                cursor_x,
                cursor_y,
            )