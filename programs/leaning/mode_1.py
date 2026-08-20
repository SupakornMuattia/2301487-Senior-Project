import math
import time
import ctypes
import cv2

from programs.skeleton import Skeleton
from programs.camera import Camera


R_ARM_LANDMARKS = [12, 14]  # shoulders, elbows, wrists
L_ARM_LANDMARKS = [11, 13]  # shoulders, elbows, wrists
R_KNEE_LANDMARKS = [26]
L_KNEE_LANDMARKS = [25]
L_HIP_LANDMARKS = [23]
R_HIP_LANDMARKS = [24]
VISIBILITY_THRESHOLD = 0.6

def check_skeleton(camera):
     if camera.landmarks:
          return True
     else:
          return False

def check_arms(camera):
     check = check_skeleton(camera)
     if check:
          pose_landmarks = camera.landmarks[0]
          right_arms = all(
               pose_landmarks[i].visibility >= VISIBILITY_THRESHOLD
               for i in R_ARM_LANDMARKS
          )
          left_arms = all(
               pose_landmarks[i].visibility >= VISIBILITY_THRESHOLD
               for i in L_ARM_LANDMARKS
          )
          if right_arms or left_arms:
               print("I see arms")
          return right_arms or left_arms
     else:
          return False
     
def calculate_angle(a, b, c):
     """Angle at vertex b (degrees) between rays b->a and b->c. Points are (x, y) pixel tuples."""
     v1 = (a[0] - b[0], a[1] - b[1])
     v2 = (c[0] - b[0], c[1] - b[1])
     mag1 = math.hypot(*v1)
     mag2 = math.hypot(*v2)
     if mag1 == 0 or mag2 == 0:
          return 0.0
     cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
     cos_angle = max(-1.0, min(1.0, cos_angle))
     return math.degrees(math.acos(cos_angle))

def check_arm_angle(camera, frame):
     check = check_skeleton(camera)
     if not check:
          return None

     pose_landmarks = camera.landmarks[0]
     h, w = frame.shape[:2]
     sides = (
          ("right", R_HIP_LANDMARKS[0], R_ARM_LANDMARKS[0], R_ARM_LANDMARKS[1], (0, 255, 255)),
          ("left", L_HIP_LANDMARKS[0], L_ARM_LANDMARKS[0], L_ARM_LANDMARKS[1], (255, 255, 0)),
     )

     angles = {}
     for side, hip_i, shoulder_i, elbow_i, color in sides:
          hip, shoulder, elbow = pose_landmarks[hip_i], pose_landmarks[shoulder_i], pose_landmarks[elbow_i]
          if min(hip.visibility, shoulder.visibility, elbow.visibility) < VISIBILITY_THRESHOLD:
               continue

          hip_pt = (int(hip.x * w), int(hip.y * h))
          shoulder_pt = (int(shoulder.x * w), int(shoulder.y * h))
          elbow_pt = (int(elbow.x * w), int(elbow.y * h))

          cv2.line(frame, hip_pt, shoulder_pt, color, 2)
          cv2.line(frame, shoulder_pt, elbow_pt, color, 2)

          angle = calculate_angle(hip_pt, shoulder_pt, elbow_pt)
          angles[side] = angle
          cv2.putText(
               frame, f"{angle:.0f} deg", (shoulder_pt[0] + 10, shoulder_pt[1]),
               cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2,
          )

     # for side, angle in angles.items():
     #      print(f"{side} shoulder angle: {angle:.1f} deg")
     return angles

def check_lean(camera, frame):
     check = check_skeleton(camera)
     if not check:
          return None

     pose_landmarks = camera.landmarks[0]
     h, w = frame.shape[:2]

     l_shoulder = pose_landmarks[L_ARM_LANDMARKS[0]]
     r_shoulder = pose_landmarks[R_ARM_LANDMARKS[0]]
     l_hip = pose_landmarks[L_HIP_LANDMARKS[0]]
     r_hip = pose_landmarks[R_HIP_LANDMARKS[0]]
     if min(l_shoulder.visibility, r_shoulder.visibility, l_hip.visibility, r_hip.visibility) < VISIBILITY_THRESHOLD:
          return None

     mid_shoulder = ((l_shoulder.x + r_shoulder.x) / 2, (l_shoulder.y + r_shoulder.y) / 2)
     mid_hip = ((l_hip.x + r_hip.x) / 2, (l_hip.y + r_hip.y) / 2)

     mid_shoulder_pt = (int(mid_shoulder[0] * w), int(mid_shoulder[1] * h))
     mid_hip_pt = (int(mid_hip[0] * w), int(mid_hip[1] * h))

     cv2.circle(frame, mid_shoulder_pt, 5, (0, 0, 255), -1)
     cv2.circle(frame, mid_hip_pt, 5, (0, 0, 255), -1)
     cv2.line(frame, mid_shoulder_pt, mid_hip_pt, (0, 0, 255), 2)

     return mid_shoulder_pt, mid_hip_pt

def guideline_x_at_y(mid_shoulder_pt, mid_hip_pt, y):
     """X coordinate of the shoulder-hip guideline (extended as a straight line) at a given y."""
     x1, y1 = mid_shoulder_pt
     x2, y2 = mid_hip_pt
     dy = y2 - y1
     if dy == 0:
          return None
     return x1 + (y - y1) * (x2 - x1) / dy

def draw_lean_guideline(frame, mid_shoulder_pt, mid_hip_pt, color=(0, 255, 255)):
     h, w = frame.shape[:2]
     top_x = guideline_x_at_y(mid_shoulder_pt, mid_hip_pt, 0)
     bottom_x = guideline_x_at_y(mid_shoulder_pt, mid_hip_pt, h)
     if top_x is None:
          return
     cv2.line(frame, (int(top_x), 0), (int(bottom_x), h), color, 1)

def _hip_sides(camera, frame, static_points):
     """Per-hip (x - guideline_x) sign at each hip's current position, keyed by 'left'/'right'.
     Returns None unless both hips are visible, since the check requires both to be tracked."""
     check = check_skeleton(camera)
     if not check or static_points is None:
          return None

     pose_landmarks = camera.landmarks[0]
     h, w = frame.shape[:2]
     mid_shoulder_pt, mid_hip_pt = static_points

     sides = {}
     for side_name, idx in (("left", L_HIP_LANDMARKS[0]), ("right", R_HIP_LANDMARKS[0])):
          hip = pose_landmarks[idx]
          if hip.visibility < VISIBILITY_THRESHOLD:
               return None
          hip_pt = (int(hip.x * w), int(hip.y * h))
          guide_x = guideline_x_at_y(mid_shoulder_pt, mid_hip_pt, hip_pt[1])
          if guide_x is None:
               return None
          sides[side_name] = 1 if hip_pt[0] - guide_x >= 0 else -1
     return sides

def capture_hip_sides(camera, frame, static_points):
     """Call once, right when the static guideline is frozen, to record each hip's starting side."""
     return _hip_sides(camera, frame, static_points)

def check_hip_behind(camera, frame, static_points, static_hip_sides):
     """True if both hips are still on their original (calibration-time) side of the static
     guideline, False if either has crossed to the other side ("pulled back"), None if the check
     isn't active (guideline not yet frozen, or a hip isn't currently visible)."""
     if static_hip_sides is None:
          return None
     current_sides = _hip_sides(camera, frame, static_points)
     if current_sides is None:
          return None
     return all(current_sides[side] == static_hip_sides[side] for side in static_hip_sides)

def draw_message(frame, text):
     h, w = frame.shape[:2]
     size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
     pt = ((w - size[0]) // 2, h // 4)
     cv2.putText(frame, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)

def draw_indicator(frame, text, seen, row=0):
     h, w = frame.shape[:2]
     color = (0, 200, 0) if seen else (0, 0, 255)
     margin = 15
     dot_radius = 20
     line_height = 100

     size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 2)
     text_x = w - margin - size[0]
     text_y = margin + size[1] + row * line_height
     dot_x = text_x - margin - dot_radius
     dot_y = text_y - size[1] // 2

     cv2.circle(frame, (dot_x, dot_y), dot_radius, color, -1)
     cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)

def draw_arms_indicator(frame, confirmed):
     draw_indicator(frame, "arms", confirmed, row=0)

def draw_legs_indicator(frame, confirmed):
     draw_indicator(frame, "legs", confirmed, row=1)

def draw_angle_indicator(frame, confirmed):
     draw_indicator(frame, "angle", confirmed, row=2)

def draw_hip_indicator(frame, confirmed):
     draw_indicator(frame, "hip", confirmed, row=3)

ARM_ANGLE_MIN = 80
ARM_ANGLE_MAX = 95

def arm_angle_in_range(angles):
     if not angles:
          return False
     return any(ARM_ANGLE_MIN <= a <= ARM_ANGLE_MAX for a in angles.values())

def check_legs(camera):
     check = check_skeleton(camera)
     if check:
          pose_landmarks = camera.landmarks[0]
          right_legs = all(
               pose_landmarks[i].visibility >= VISIBILITY_THRESHOLD
               for i in R_KNEE_LANDMARKS
          )
          left_legs = all(
               pose_landmarks[i].visibility >= VISIBILITY_THRESHOLD
               for i in L_KNEE_LANDMARKS
          )
          if right_legs or left_legs:
               print("I see legs")
          return right_legs or left_legs
     else:
          return False

def get_screen_size():
     """Primary monitor resolution in pixels (Windows)."""
     user32 = ctypes.windll.user32
     return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def fit_window_to_screen(window_name, frame, margin=0.9):
     """Resize an existing WINDOW_NORMAL window so frame fills as much of the
     screen as possible (up to `margin`) while keeping its aspect ratio."""
     h, w = frame.shape[:2]
     screen_w, screen_h = Camera.get_screen_size()
     scale = min(screen_w * margin / w, screen_h * margin / h)
     cv2.resizeWindow(window_name, int(w * scale), int(h * scale))

if __name__ == "__main__":
     camera = Camera(camera=0, crop_w=405, crop_h=720)
     skeleton = Skeleton(camera)

     state = "wait_arms"
     countdown_start = None
     static_points = None
     static_hip_sides = None

     camera.open_camera()
     try:
          while True:
               ok, frame = camera.cap.read()
               window_name = "Camera Preview (q or Esc to quit)"
               cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
               window_sized = False
               if not ok:
                    print("Failed to read frame from camera")
                    break

               frame = cv2.flip(frame, 1)
               # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
               if not window_sized:
                    fit_window_to_screen(window_name, frame)
                    window_sized = True
               # frame = Camera.frame_crop(frame, camera.crop_w, camera.crop_h)

               result = skeleton.detect(frame)
               if skeleton.draw_enabled and result is not None and result.pose_landmarks:
                    frame = skeleton.draw_landmarks(frame, result)

               cv2.putText(
                    frame, f"smoothing: {'on' if skeleton.smoothing_enabled else 'off'} (m to toggle)",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 255) if skeleton.smoothing_enabled else (0, 0, 255), 2,
               )

               arm_angles = check_arm_angle(camera, frame)
               lean_points = check_lean(camera, frame)
               hip_ok = check_hip_behind(camera, frame, static_points, static_hip_sides)

               if state == "wait_arms":
                    draw_message(frame, "see your arms")
                    if check_arms(camera):
                         state = "countdown_arms"
                         countdown_start = time.time()
               elif state == "countdown_arms":
                    if not check_arms(camera):
                         state = "wait_arms"
                    else:
                         remaining = 3 - int(time.time() - countdown_start)
                         if remaining <= 0:
                              state = "wait_legs"
                         else:
                              draw_message(frame, str(remaining))
               elif state == "wait_legs":
                    if not check_arms(camera):
                         state = "wait_arms"
                    else:
                         draw_message(frame, "see your legs")
                         if check_legs(camera):
                              state = "countdown_legs"
                              countdown_start = time.time()
               elif state == "countdown_legs":
                    if not check_arms(camera):
                         state = "wait_arms"
                    elif not check_legs(camera):
                         state = "wait_legs"
                    else:
                         remaining = 3 - int(time.time() - countdown_start)
                         if remaining <= 0:
                              state = "wait_angle"
                         else:
                              draw_message(frame, str(remaining))
               elif state == "wait_angle":
                    if not check_arms(camera):
                         state = "wait_arms"
                    elif not check_legs(camera):
                         state = "wait_legs"
                    else:
                         draw_message(frame, "hold arm at 90 degrees")
                         if arm_angle_in_range(arm_angles):
                              state = "countdown_angle"
                              countdown_start = time.time()
               elif state == "countdown_angle":
                    if not check_arms(camera):
                         state = "wait_arms"
                    elif not check_legs(camera):
                         state = "wait_legs"
                    elif not arm_angle_in_range(arm_angles):
                         state = "wait_angle"
                    else:
                         remaining = 3 - int(time.time() - countdown_start)
                         if remaining <= 0:
                              state = "done"
                              static_points = lean_points
                              static_hip_sides = capture_hip_sides(camera, frame, static_points)
                         else:
                              draw_message(frame, str(remaining))
               elif state == "done":
                    if static_points is not None:
                         draw_lean_guideline(frame, *static_points, color=(0, 255, 255))
                    if lean_points is not None:
                         draw_lean_guideline(frame, *lean_points, color=(255, 0, 255))
                    if hip_ok is False:
                         draw_message(frame, "hip pulled back")

               arms_confirmed = state not in ("wait_arms", "countdown_arms")
               legs_confirmed = state not in ("wait_arms", "countdown_arms", "wait_legs", "countdown_legs")
               angle_confirmed = state == "done"
               hip_confirmed = state == "done" and hip_ok is True
               draw_arms_indicator(frame, arms_confirmed)
               draw_legs_indicator(frame, legs_confirmed)
               draw_angle_indicator(frame, angle_confirmed)
               draw_hip_indicator(frame, hip_confirmed)

               cv2.imshow(window_name, frame)
               key = cv2.waitKey(1) & 0xFF
               if key in (ord("q"), 27):  # 27 = Esc
                    break
               elif key == ord("t"):  # toggle landmark drawing
                    skeleton.disable() if skeleton.draw_enabled else skeleton.enable()
               elif key == ord("m"):  # toggle median+EMA smoothing
                    skeleton.toggle_smoothing()
     finally:
          skeleton.landmarker.close()
          camera.release()
