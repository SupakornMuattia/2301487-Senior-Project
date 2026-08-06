import time

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from camera import Camera

MODEL_PATH = "model\pose_landmarker_full.task"


class Skeleton:
    def __init__(self, model_path: str = MODEL_PATH):
        self.camera = Camera(0, 405, 720)

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    @staticmethod
    def draw_landmarks(frame, detection_result):
        for pose_landmarks in detection_result.pose_landmarks:
            landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in pose_landmarks
                ]
            )
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                landmarks_proto,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return frame

    def run_preview(self):
        self.camera.open_camera()
        window_name = "Skeleton Preview (q or Esc to quit)"
        try:
            while True:
                ok, frame = self.camera.cap.read()
                if not ok:
                    print("Failed to read frame from camera")
                    break

                frame = cv2.flip(frame, 1)
                frame = Camera.frame_crop(frame, self.camera.crop_w, self.camera.crop_h)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.time() * 1000)

                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    frame = self.draw_landmarks(frame, result)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 27 = Esc
                    break
        finally:
            self.landmarker.close()
            self.camera.release()


def main() -> None:
    skeleton = Skeleton()
    skeleton.run_preview()


if __name__ == "__main__":
    main()
