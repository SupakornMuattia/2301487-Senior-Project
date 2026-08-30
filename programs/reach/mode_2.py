import math
import time
import cv2

from programs.assets.camera import Camera
from programs.assets.skeleton import Skeleton
from programs.checks.arms import check_arms, R_ARM_LANDMARKS, L_ARM_LANDMARKS
from programs.checks.legs import check_legs, check_skeleton, R_LEGS_LANDMARKS, L_LEGS_LANDMARKS, VISIBILITY_THRESHOLD
from programs.checks.facing import facing_camera_live, facing_sideways_live

CHECK_HOLD_SECONDS = 3  # countdown shown once arms/legs/facing all pass
ARM_ANGLE_MIN = 80  # hip-shoulder-elbow angle counted as "raised to 90 degrees"
ARM_ANGLE_MAX = 95

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

GUIDE_MARK_X_FRAC = 0.22  # horizontal position of the guide mark, as a fraction of frame width
GUIDE_MARK_THICKNESS = 3
MARK_TOLERANCE_FRAC = 0.06  # allowed horizontal drift (fraction of frame width) from the guide mark


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


def check_distance(skeleton, camera, frame):
     face_result = skeleton.detect_face(frame)
     if skeleton.draw_enabled and face_result is not None and face_result.face_landmarks:
          skeleton.draw_face_landmarks(frame, face_result)
     ipd_px = skeleton.get_ipd_px(camera.face_landmarks, frame.shape)
     return skeleton.get_distance_cm(ipd_px)


def draw_guide_mark(frame):
     """Draw a white-outline person mark on the left side of the frame,
     guiding the user where to stand for state 2."""
     h, w = frame.shape[:2]
     cx = int(w * GUIDE_MARK_X_FRAC)

     head_center = (cx, int(h * 0.16))
     head_radius = int(h * 0.06)
     neck = (cx, int(h * 0.23))
     shoulder_l = (cx - int(w * 0.09), int(h * 0.27))
     shoulder_r = (cx + int(w * 0.09), int(h * 0.27))
     elbow_l = (cx - int(w * 0.13), int(h * 0.40))
     elbow_r = (cx + int(w * 0.13), int(h * 0.40))
     hand_l = (cx - int(w * 0.11), int(h * 0.53))
     hand_r = (cx + int(w * 0.11), int(h * 0.53))
     hip_l = (cx - int(w * 0.06), int(h * 0.55))
     hip_r = (cx + int(w * 0.06), int(h * 0.55))
     knee_l = (cx - int(w * 0.07), int(h * 0.75))
     knee_r = (cx + int(w * 0.07), int(h * 0.75))
     foot_l = (cx - int(w * 0.08), int(h * 0.92))
     foot_r = (cx + int(w * 0.08), int(h * 0.92))

     cv2.circle(frame, head_center, head_radius, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, neck, hip_l, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, neck, hip_r, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, shoulder_l, shoulder_r, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, shoulder_l, elbow_l, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, elbow_l, hand_l, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, shoulder_r, elbow_r, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, elbow_r, hand_r, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, hip_l, hip_r, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, hip_l, knee_l, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, knee_l, foot_l, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, hip_r, knee_r, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, knee_r, foot_r, WHITE, GUIDE_MARK_THICKNESS)


def draw_guide_mark_sideways(frame):
     """Draw a white-outline person mark, viewed from the side, on the left
     side of the frame, guiding the user to turn side-on to the camera and
     raise their arm to 90 degrees for state 3."""
     h, w = frame.shape[:2]
     cx = int(w * GUIDE_MARK_X_FRAC)

     head_center = (cx, int(h * 0.16))
     head_radius = int(h * 0.06)
     face_tip = (cx + int(head_radius * 1.4), int(h * 0.17))
     neck = (cx, int(h * 0.23))
     shoulder = (cx, int(h * 0.27))
     elbow = (cx + int(w * 0.13), int(h * 0.27))  # raised forward to shoulder height (90 deg from torso)
     hand = (cx + int(w * 0.24), int(h * 0.27))
     hip = (cx + int(w * 0.02), int(h * 0.55))
     knee = (cx + int(w * 0.03), int(h * 0.75))
     ankle = (cx + int(w * 0.03), int(h * 0.90))
     foot_tip = (cx + int(w * 0.15), int(h * 0.93))

     cv2.circle(frame, head_center, head_radius, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, head_center, face_tip, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, neck, hip, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, shoulder, elbow, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, elbow, hand, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, hip, knee, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, knee, ankle, WHITE, GUIDE_MARK_THICKNESS)
     cv2.line(frame, ankle, foot_tip, WHITE, GUIDE_MARK_THICKNESS)


def check_on_mark(camera, frame, mark_x_frac=GUIDE_MARK_X_FRAC, tolerance_frac=MARK_TOLERANCE_FRAC):
     """Whether the user's feet are horizontally aligned with the guide mark."""
     if not check_skeleton(camera):
          return False
     pose_landmarks = camera.landmarks[0]
     r_ankle = pose_landmarks[R_LEGS_LANDMARKS[2]]
     l_ankle = pose_landmarks[L_LEGS_LANDMARKS[2]]
     if min(r_ankle.visibility, l_ankle.visibility) < VISIBILITY_THRESHOLD:
          return False
     w = frame.shape[1]
     feet_x = (r_ankle.x + l_ankle.x) / 2 * w
     mark_x = w * mark_x_frac
     return abs(feet_x - mark_x) <= w * tolerance_frac


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
     """Hip-shoulder-elbow angle for each arm that's visible, keyed by side.
     Draws the hip-shoulder-elbow lines and the angle in degrees next to
     each shoulder, for debugging."""
     if not check_skeleton(camera):
          return {}
     pose_landmarks = camera.landmarks[0]
     h, w = frame.shape[:2]
     sides = (
          ("right", R_LEGS_LANDMARKS[0], R_ARM_LANDMARKS[0], R_ARM_LANDMARKS[1], (0, 255, 255)),
          ("left", L_LEGS_LANDMARKS[0], L_ARM_LANDMARKS[0], L_ARM_LANDMARKS[1], (255, 255, 0)),
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
     return angles


def arm_angle_in_range(angles, angle_min=ARM_ANGLE_MIN, angle_max=ARM_ANGLE_MAX):
     """Whether at least one tracked arm is raised to ~90 degrees from the torso."""
     return any(angle_min <= a <= angle_max for a in angles.values())


def draw_message(frame, text):
     h, w = frame.shape[:2]
     size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
     pt = ((w - size[0]) // 2, h // 4)
     cv2.putText(frame, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)


if __name__ == "__main__":
     camera = Camera(camera=0, crop_w=720, crop_h=1280)
     skeleton = Skeleton(camera)

     state = "state_1"
     countdown_start = None
     captured_distance_cm = None

     camera.open_camera()
     window_name = "Mode 2 (q or Esc to quit)"
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

               if state == "state_1":
                    arms_ok = check_arms(camera)
                    legs_ok = check_legs(camera)
                    facing_ok = facing_camera_live(camera)

                    distance_cm = check_distance(skeleton, camera, frame)
                    if distance_cm is not None:
                         captured_distance_cm = distance_cm

                    draw_status(frame, "arms", arms_ok, row=0)
                    draw_status(frame, "legs", legs_ok, row=1)
                    draw_status(frame, "facing", facing_ok, row=2)

                    if arms_ok and legs_ok and facing_ok:
                         state = "state_1_countdown"
                         countdown_start = time.time()
                    else:
                         draw_message(frame, "show arms, legs and face \nthe camera")
               elif state == "state_1_countdown":
                    arms_ok = check_arms(camera)
                    legs_ok = check_legs(camera)
                    facing_ok = facing_camera_live(camera)

                    distance_cm = check_distance(skeleton, camera, frame)
                    if distance_cm is not None:
                         captured_distance_cm = distance_cm

                    draw_status(frame, "arms", arms_ok, row=0)
                    draw_status(frame, "legs", legs_ok, row=1)
                    draw_status(frame, "facing", facing_ok, row=2)

                    if not (arms_ok and legs_ok and facing_ok):
                         state = "state_1"
                         countdown_start = None
                    else:
                         remaining = CHECK_HOLD_SECONDS - int(time.time() - countdown_start)
                         if remaining <= 0:
                              state = "state_2"
                              countdown_start = None
                         else:
                              draw_message(frame, str(remaining))
               elif state == "state_2":
                    draw_guide_mark(frame)
                    on_mark = check_on_mark(camera, frame)

                    if on_mark:
                         if countdown_start is None:
                              countdown_start = time.time()
                         remaining = CHECK_HOLD_SECONDS - int(time.time() - countdown_start)
                         if remaining <= 0:
                              state = "state_3"
                              countdown_start = None
                         else:
                              draw_message(frame, str(remaining))
                    else:
                         countdown_start = None
                         draw_message(frame, "stand on the mark")
               elif state == "state_3":
                    draw_guide_mark_sideways(frame)
                    sideways_ok = facing_sideways_live(camera)
                    arm_angles = check_arm_angle(camera, frame)
                    arm_ok = arm_angle_in_range(arm_angles)

                    draw_status(frame, "sideways", sideways_ok, row=0)
                    draw_status(frame, "arm at 90 deg", arm_ok, row=1)

                    if sideways_ok and arm_ok:
                         if countdown_start is None:
                              countdown_start = time.time()
                         remaining = CHECK_HOLD_SECONDS - int(time.time() - countdown_start)
                         if remaining <= 0:
                              state = "state_4"
                              countdown_start = None
                         else:
                              draw_message(frame, str(remaining))
                    else:
                         countdown_start = None
                         draw_message(frame, "turn sideways \nand \nraise your arm to 90 degrees")
               elif state == "state_4":
                    draw_message(frame, "wait state 4")

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
          skeleton.face_landmarker.close()
          camera.release()
