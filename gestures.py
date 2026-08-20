import pyautogui

from config import (
    CURSOR_HOLD_TIME,
    CURSOR_RELEASE_FRAMES,
    MIC_HOLD_TIME,
    SEND_HOLD_TIME,
    MIC_RELEASE_FRAMES,
    COMMAND_RELEASE_FRAMES,
    MIC_GLOBAL_COOLDOWN,
)


class GestureController:
    def __init__(self):
        # Right Thumb Down -> cursor toggle
        self.cursor_hold_start = None
        self.cursor_latched = False
        self.cursor_release_frames = 0

        # Fist -> microphone, independently tracked for each hand
        self.mic_hold_start = {
            "Left": None,
            "Right": None,
        }

        self.mic_latched = {
            "Left": False,
            "Right": False,
        }

        self.mic_release_frames = {
            "Left": 0,
            "Right": 0,
        }

        self.mic_block_until = 0.0

        # Open Palm -> send, independently tracked for each hand
        self.send_hold_start = {
            "Left": None,
            "Right": None,
        }

        self.send_latched = {
            "Left": False,
            "Right": False,
        }

        self.send_release_frames = {
            "Left": 0,
            "Right": 0,
        }

    def process(
        self,
        gesture,
        frame_time,
        cursor_enabled,
        screen_width,
        screen_height,
    ):
        # Right Thumb Down -> cursor toggle
        if gesture == "Thumb_Down":
            self.cursor_release_frames = 0

            if self.cursor_hold_start is None:
                self.cursor_hold_start = frame_time

            elif (
                not self.cursor_latched
                and frame_time - self.cursor_hold_start
                >= CURSOR_HOLD_TIME
            ):
                cursor_enabled = not cursor_enabled
                self.cursor_latched = True

                if cursor_enabled:
                    pyautogui.moveTo(
                        screen_width // 2,
                        screen_height // 2,
                    )

                print(
                    "CURSOR ON"
                    if cursor_enabled
                    else "CURSOR OFF"
                )

        else:
            self.cursor_hold_start = None

            if self.cursor_latched:
                self.cursor_release_frames += 1

                if (
                    self.cursor_release_frames
                    >= CURSOR_RELEASE_FRAMES
                ):
                    self.cursor_latched = False
                    self.cursor_release_frames = 0

        return cursor_enabled

    def process_mic(
        self,
        handedness,
        gesture,
        frame_time,
        cursor_enabled,
    ):
        # Disable semantic gestures while cursor mode is ON
        if cursor_enabled:
            self.mic_hold_start[handedness] = None
            self.mic_latched[handedness] = False
            self.mic_release_frames[handedness] = 0
            return

        # Closed Fist -> microphone
        if gesture == "Closed_Fist":
            self.mic_release_frames[handedness] = 0

            if self.mic_hold_start[handedness] is None:
                self.mic_hold_start[handedness] = frame_time

            elif (
                not self.mic_latched[handedness]
                and frame_time - self.mic_hold_start[handedness]
                >= MIC_HOLD_TIME
                and frame_time >= self.mic_block_until
            ):
                pyautogui.hotkey(
                    "ctrl",
                    "shift",
                    "d",
                )

                self.mic_latched[handedness] = True

                self.mic_block_until = (
                    frame_time + MIC_GLOBAL_COOLDOWN
                )

                print(f"MIC TOGGLE ({handedness})")

        else:
            self.mic_hold_start[handedness] = None

            if self.mic_latched[handedness]:
                self.mic_release_frames[handedness] += 1

                if (
                    self.mic_release_frames[handedness]
                    >= MIC_RELEASE_FRAMES
                ):
                    self.mic_latched[handedness] = False
                    self.mic_release_frames[handedness] = 0

    def process_send(
        self,
        handedness,
        gesture,
        frame_time,
        cursor_enabled,
    ):
        # Disable semantic gestures while cursor mode is ON
        if cursor_enabled:
            self.send_hold_start[handedness] = None
            self.send_latched[handedness] = False
            self.send_release_frames[handedness] = 0
            return

        # Open Palm -> Enter
        if gesture == "Open_Palm":
            self.send_release_frames[handedness] = 0

            if self.send_hold_start[handedness] is None:
                self.send_hold_start[handedness] = frame_time

            elif (
                not self.send_latched[handedness]
                and frame_time - self.send_hold_start[handedness]
                >= SEND_HOLD_TIME
            ):
                pyautogui.press("enter")

                self.send_latched[handedness] = True

                print(f"SEND ({handedness})")

        else:
            self.send_hold_start[handedness] = None

            if self.send_latched[handedness]:
                self.send_release_frames[handedness] += 1

                if (
                    self.send_release_frames[handedness]
                    >= COMMAND_RELEASE_FRAMES
                ):
                    self.send_latched[handedness] = False
                    self.send_release_frames[handedness] = 0