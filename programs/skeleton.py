import math
import time
import cv2
import ctypes

from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarksConnections,
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
    drawing_styles,
    drawing_utils,
)

from programs.camera import Camera
from programs.calibration import Calibration

MODEL_PATH = "model/pose_landmarker_full.task"
FACE_MODEL_PATH = "model/face_landmarker.task"

RIGHT_IRIS_CENTER = 468  # center points of MediaPipe's 478-point face mesh iris rings
LEFT_IRIS_CENTER = 473

REAL_IPD_CM = 6.3  # average adult interpupillary distance

class Skeleton:
    def __init__(
        self,
        camera: Camera = None,
        model_path: str = MODEL_PATH,
        face_model_path: str = FACE_MODEL_PATH,
        frame_width: int = 1280,
        real_ipd_cm: float = REAL_IPD_CM,
        calibration_facing: str = "front",
    ):
        self.camera = camera if camera is not None else Camera(0, 720, 1280)
        self.draw_enabled = True
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
        )
        self.landmarker = PoseLandmarker.create_from_options(self.options)
        self.face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
        )
        self.face_landmarker = FaceLandmarker.create_from_options(self.face_options)

        self.real_ipd_cm = real_ipd_cm
        try:
            self.focal_length_px = Calibration(facing=calibration_facing).focal_length_px(
                image_width_px=frame_width
            )
        except (RuntimeError, ValueError) as e:
            print(f"Calibration unavailable ({e}); distance_cm will be disabled")
            self.focal_length_px = None

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

    @staticmethod
    def draw_face_landmarks(frame, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        for face_landmarks in face_landmarks_list:
            drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
            )
            drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
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

    def detect_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        self.camera.face_landmarks = result.face_landmarks if result.face_landmarks else None
        return result

    @staticmethod
    def get_ipd_px(face_landmarks, frame_shape):
        """Pixel distance between the left/right iris centers, or None if no
        face (or an unrefined face mesh without iris landmarks) is available."""
        if not face_landmarks:
            return None
        landmarks = face_landmarks[0]
        if len(landmarks) <= LEFT_IRIS_CENTER:
            return None
        h, w = frame_shape[:2]
        right = landmarks[RIGHT_IRIS_CENTER]
        left = landmarks[LEFT_IRIS_CENTER]
        return math.hypot((left.x - right.x) * w, (left.y - right.y) * h)

    def get_distance_cm(self, ipd_px):
        """Camera-to-face distance (cm) via the pinhole model:
        distance_cm = (real_ipd_cm * focal_length_px) / ipd_pixel_distance."""
        if not ipd_px or self.focal_length_px is None:
            return None
        return (self.real_ipd_cm * self.focal_length_px) / ipd_px

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
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                # frame = Camera.frame_crop(frame, self.camera.crop_w, self.camera.crop_h)
                
                if not window_sized:
                    self.fit_window_to_screen(window_name, frame)
                    window_sized = True

                result = self.detect(frame)
                face_result = self.detect_face(frame)

                if self.draw_enabled and result is not None and result.pose_landmarks:
                    frame = self.draw_landmarks(frame, result)
                if self.draw_enabled and face_result is not None and face_result.face_landmarks:
                    frame = self.draw_face_landmarks(frame, face_result)

                ipd_px = self.get_ipd_px(self.camera.face_landmarks, frame.shape)
                if ipd_px is not None:
                    cv2.putText(
                        frame, f"ipd: {ipd_px:.1f}px", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    )
                    distance_cm = self.get_distance_cm(ipd_px)
                    if distance_cm is not None:
                        cv2.putText(
                            frame, f"distance: {distance_cm:.1f}cm", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 27 = Esc
                    break
        finally:
            self.landmarker.close()
            self.face_landmarker.close()
            self.camera.release()

if __name__ == "__main__":
    skeleton = Skeleton()
    skeleton.run_preview()
