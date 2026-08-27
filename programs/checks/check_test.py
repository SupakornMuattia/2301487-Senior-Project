import cv2

from programs.assets.camera import Camera
from programs.assets.skeleton import Skeleton
from programs.checks.arms import check_arms
from programs.checks.legs import check_legs
from programs.checks.facing import shoulder_hip_ratio, facing_camera

GREEN = (0, 200, 0)
RED = (0, 0, 255)


def draw_status(frame, text, ok, row):
     margin = 15
     line_height = 50
     dot_radius = 10
     color = GREEN if ok else RED
     y = margin + dot_radius + row * line_height
     cv2.circle(frame, (margin + dot_radius, y), dot_radius, color, -1)
     cv2.putText(
          frame, text, (margin + 2 * dot_radius + 10, y + dot_radius),
          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
     )


if __name__ == "__main__":
     camera = Camera(camera=0, crop_w=720, crop_h=1280)
     skeleton = Skeleton(camera)

     baseline_ratio = None

     camera.open_camera()
     window_name = "Check Test (q or Esc to quit, c to calibrate facing)"
     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
     window_sized = False
     try:
          while True:
               ok, frame = camera.cap.read()
               if not ok:
                    print("Failed to read frame from camera")
                    break

               frame = cv2.flip(frame, 1)
               if not window_sized:
                    Camera.fit_window_to_screen(window_name, frame)
                    window_sized = True

               result = skeleton.detect(frame)
               if skeleton.draw_enabled and result is not None and result.pose_landmarks:
                    frame = skeleton.draw_landmarks(frame, result)

               arms_ok = check_arms(camera)
               legs_ok = check_legs(camera)
               ratio = shoulder_hip_ratio(camera, frame)

               draw_status(frame, "arms", arms_ok, row=0)
               draw_status(frame, "legs", legs_ok, row=1)
               if baseline_ratio is None:
                    draw_status(frame, "facing (press c to calibrate)", False, row=2)
               else:
                    draw_status(frame, "facing", facing_camera(ratio, baseline_ratio), row=2)

               cv2.imshow(window_name, frame)
               key = cv2.waitKey(1) & 0xFF
               if key in (ord("q"), 27):  # 27 = Esc
                    break
               elif key == ord("c"):  # (re)capture the facing baseline
                    if ratio is not None:
                         baseline_ratio = ratio
                         print(f"Facing baseline captured: {baseline_ratio:.3f}")
                    else:
                         print("Can't capture facing baseline - shoulders/hips not visible")
     finally:
          skeleton.landmarker.close()
          skeleton.face_landmarker.close()
          camera.release()
