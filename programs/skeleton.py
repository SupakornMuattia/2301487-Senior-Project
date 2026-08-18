import time
import cv2
import ctypes

from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
    drawing_styles,
    drawing_utils,
)

from programs.camera import Camera

MODEL_PATH = "model/pose_landmarker_full.task"

class Skeleton:
    def __init__(self, camera: Camera = None, model_path: str = MODEL_PATH):
        self.camera = camera if camera is not None else Camera(0, 405, 720)
        self.draw_enabled = True
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
        )
        self.landmarker = PoseLandmarker.create_from_options(self.options)

    @staticmethod
    def draw_landmarks(frame, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        for pose_landmarks in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                frame,
                pose_landmarks,
                PoseLandmarksConnections.POSE_LANDMARKS,
                drawing_styles.get_default_pose_landmarks_style(),
            )
        return frame

    def enable(self):
        self.draw_enabled = True

    def disable(self):
        self.draw_enabled = False

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        self.camera.landmarks = result.pose_landmarks if result.pose_landmarks else None
        return result

    @staticmethod
    def get_screen_size():
        """Primary monitor resolution in pixels (Windows)."""
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    
    @staticmethod
    def fit_window_to_screen(window_name, frame, margin=0.9):
        """Resize an existing WINDOW_NORMAL window so frame fills as much of the
        screen as possible (up to `margin`) while keeping its aspect ratio."""
        h, w = frame.shape[:2]
        screen_w, screen_h = Camera.get_screen_size()
        scale = min(screen_w * margin / w, screen_h * margin / h)
        cv2.resizeWindow(window_name, int(w * scale), int(h * scale))

    def run_preview(self):
        self.camera.open_camera()
        window_name = "Skeleton Preview (q or Esc to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_sized = False
        try:
            while True:
                ok, frame = self.camera.cap.read()
                if not ok:
                    print("Failed to read frame from camera")
                    break

                frame = cv2.flip(frame, 1)
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                # frame = Camera.frame_crop(frame, self.camera.crop_w, self.camera.crop_h)
                
                if not window_sized:
                    self.fit_window_to_screen(window_name, frame)
                    window_sized = True

                result = self.detect(frame)

                if self.draw_enabled and result is not None and result.pose_landmarks:
                    frame = self.draw_landmarks(frame, result)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 27 = Esc
                    break
        finally:
            self.landmarker.close()
            self.camera.release()

if __name__ == "__main__":
    skeleton = Skeleton()
    skeleton.run_preview()
