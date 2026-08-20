import math
import time
import cv2
import ctypes
from collections import deque

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

MEDIAN_WINDOW = 5  # frames considered per landmark when rejecting outlier spikes
EMA_ALPHA = 0.3  # higher = more responsive/less smooth, lower = smoother/more lag


class SmoothedLandmark:
    """Duck-types a MediaPipe NormalizedLandmark (x/y/z/visibility/presence) so
    smoothed output is a drop-in replacement wherever raw landmarks are
    consumed, including mediapipe's own drawing_utils."""

    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x, y, z, visibility, presence):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


class LandmarkSmoother:
    """Per-landmark median filter (rejects single-frame outlier spikes) feeding
    into an exponential moving average (smooths the remaining jitter)."""

    def __init__(self, median_window: int = MEDIAN_WINDOW, alpha: float = EMA_ALPHA):
        if median_window < 1:
            raise ValueError("median_window must be >= 1")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.median_window = median_window
        self.alpha = alpha
        self.history = None  # list[deque] of (x, y, z, visibility, presence), one deque per landmark
        self.ema = None  # list[x, y, z, visibility, presence], one per landmark

    def reset(self):
        """Drop buffered frames, e.g. when tracking is lost, so the next
        detection isn't blended against stale positions."""
        self.history = None
        self.ema = None

    @staticmethod
    def _median(values):
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    def smooth(self, pose_landmarks):
        """pose_landmarks: a single pose's landmark list (each item exposing
        .x/.y/.z/.visibility), as in `camera.landmarks[0]`. Returns a new list
        of the same length, median-filtered then EMA-smoothed, or None if no
        landmarks were given."""
        if not pose_landmarks:
            return None

        count = len(pose_landmarks)
        if self.history is None or len(self.history) != count:
            self.history = [deque(maxlen=self.median_window) for _ in range(count)]
            self.ema = [None] * count

        smoothed = []
        for i, lm in enumerate(pose_landmarks):
            presence = lm.presence if lm.presence is not None else 1.0
            self.history[i].append((lm.x, lm.y, lm.z, lm.visibility, presence))
            med_x = self._median(v[0] for v in self.history[i])
            med_y = self._median(v[1] for v in self.history[i])
            med_z = self._median(v[2] for v in self.history[i])
            med_v = self._median(v[3] for v in self.history[i])
            med_p = self._median(v[4] for v in self.history[i])

            if self.ema[i] is None:
                self.ema[i] = [med_x, med_y, med_z, med_v, med_p]
            else:
                prev = self.ema[i]
                a = self.alpha
                self.ema[i] = [
                    a * med_x + (1 - a) * prev[0],
                    a * med_y + (1 - a) * prev[1],
                    a * med_z + (1 - a) * prev[2],
                    a * med_v + (1 - a) * prev[3],
                    a * med_p + (1 - a) * prev[4],
                ]
            smoothed.append(SmoothedLandmark(*self.ema[i]))
        return smoothed


class Skeleton:
    def __init__(
        self,
        camera: Camera = None,
        model_path: str = MODEL_PATH,
        face_model_path: str = FACE_MODEL_PATH,
        frame_width: int = 1280,
        real_ipd_cm: float = REAL_IPD_CM,
        calibration_facing: str = "front",
        median_window: int = MEDIAN_WINDOW,
        ema_alpha: float = EMA_ALPHA,
    ):
        self.camera = camera if camera is not None else Camera(0, 720, 1280)
        self.draw_enabled = True
        self.smoothing_enabled = True
        self.landmark_smoother = LandmarkSmoother(median_window, ema_alpha)
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

    SMOOTHED_DRAWING_SPEC = drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)

    def draw_smoothed_landmarks(self, frame):
        """Draws camera.landmarks (post median-filter + EMA) in magenta, so it
        can be visually compared against the raw (green) overlay from
        draw_landmarks while tuning median_window/ema_alpha."""
        if not self.camera.landmarks:
            return frame
        for pose_landmarks in self.camera.landmarks:
            drawing_utils.draw_landmarks(
                frame,
                pose_landmarks,
                PoseLandmarksConnections.POSE_LANDMARKS,
                self.SMOOTHED_DRAWING_SPEC,
                self.SMOOTHED_DRAWING_SPEC,
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

    def enable_smoothing(self):
        self.smoothing_enabled = True

    def disable_smoothing(self):
        self.smoothing_enabled = False
        self.landmark_smoother.reset()

    def toggle_smoothing(self):
        if self.smoothing_enabled:
            self.disable_smoothing()
        else:
            self.enable_smoothing()

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            self.landmark_smoother.reset()
            self.camera.landmarks = None
        elif self.smoothing_enabled:
            self.camera.landmarks = [self.landmark_smoother.smooth(result.pose_landmarks[0])]
        else:
            self.landmark_smoother.reset()
            self.camera.landmarks = result.pose_landmarks
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
                    if self.smoothing_enabled:
                        frame = self.draw_smoothed_landmarks(frame)
                if self.draw_enabled and face_result is not None and face_result.face_landmarks:
                    frame = self.draw_face_landmarks(frame, face_result)

                cv2.putText(
                    frame, f"smoothing: {'on' if self.smoothing_enabled else 'off'} (m to toggle)",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 255) if self.smoothing_enabled else (0, 0, 255), 2,
                )

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
                elif key == ord("m"):  # toggle median+EMA smoothing
                    self.toggle_smoothing()
        finally:
            self.landmarker.close()
            self.face_landmarker.close()
            self.camera.release()

if __name__ == "__main__":
    skeleton = Skeleton()
    skeleton.run_preview()
