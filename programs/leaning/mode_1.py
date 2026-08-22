import math
import time
import ctypes
import cv2

from programs.skeleton import Skeleton
from programs.camera import Camera


R_ARM_LANDMARKS = [12, 14, 16]  # shoulder, elbow, wrist
L_ARM_LANDMARKS = [11, 13, 15]  # shoulder, elbow, wrist
R_WRIST_LANDMARK = R_ARM_LANDMARKS[2]  # right wrist
L_WRIST_LANDMARK = L_ARM_LANDMARKS[2]  # left wrist
R_LEGS_LANDMARKS = [24, 26, 28]  # hip, knee, ankle
L_LEGS_LANDMARKS = [23, 25, 27]  # hip, knee, ankle
R_HIP_LANDMARKS = R_LEGS_LANDMARKS[0]  # right hip
L_HIP_LANDMARKS = L_LEGS_LANDMARKS[0] # left hip
VISIBILITY_THRESHOLD = 0.6
DISTANCE_MIN_CM = 100
DISTANCE_MAX_CM = 200
DISTANCE_STILLNESS_TOLERANCE_CM = 10  # max drift from the anchor reading allowed during the hold

def check_distance(skeleton, camera, frame):
     face_result = skeleton.detect_face(frame)
     if skeleton.draw_enabled and face_result is not None and face_result.face_landmarks:
          skeleton.draw_face_landmarks(frame, face_result)
     ipd_px = skeleton.get_ipd_px(camera.face_landmarks, frame.shape)
     return skeleton.get_distance_cm(ipd_px)

def distance_in_range(distance_cm):
     return distance_cm is not None and DISTANCE_MIN_CM <= distance_cm <= DISTANCE_MAX_CM

def distance_still(distance_cm, anchor_distance_cm):
     return (
          distance_cm is not None
          and anchor_distance_cm is not None
          and abs(distance_cm - anchor_distance_cm) <= DISTANCE_STILLNESS_TOLERANCE_CM
     )

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
          ("right", R_HIP_LANDMARKS, R_ARM_LANDMARKS[0], R_ARM_LANDMARKS[1], (0, 255, 255)),
          ("left", L_HIP_LANDMARKS, L_ARM_LANDMARKS[0], L_ARM_LANDMARKS[1], (255, 255, 0)),
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

def check_lean(camera, frame):
     check = check_skeleton(camera)
     if not check:
          return None

     pose_landmarks = camera.landmarks[0]
     h, w = frame.shape[:2]

     l_shoulder = pose_landmarks[L_ARM_LANDMARKS[0]]
     r_shoulder = pose_landmarks[R_ARM_LANDMARKS[0]]
     l_hip = pose_landmarks[L_HIP_LANDMARKS]
     r_hip = pose_landmarks[R_HIP_LANDMARKS]
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
     cv2.line(frame, (int(top_x), 0), (int(bottom_x), h), color, 10)

def draw_message(frame, text):
     h, w = frame.shape[:2]
     size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
     pt = ((w - size[0]) // 2, h // 4)
     cv2.putText(frame, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

def draw_indicator(frame, text, seen, row=0):
     h, w = frame.shape[:2]
     color = (0, 200, 0) if seen else (0, 0, 255)
     margin = 15
     dot_radius = 10
     line_height = 50

     size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
     text_x = w - margin - size[0]
     text_y = margin + size[1] + row * line_height
     dot_x = text_x - margin - dot_radius
     dot_y = text_y - size[1] // 2

     cv2.circle(frame, (dot_x, dot_y), dot_radius, color, -1)
     cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

def draw_distance_indicator(frame, confirmed):
     draw_indicator(frame, "distance", confirmed, row=0)

def draw_arms_indicator(frame, confirmed):
     draw_indicator(frame, "arms", confirmed, row=1)

def draw_legs_indicator(frame, confirmed):
     draw_indicator(frame, "legs", confirmed, row=2)

def draw_angle_indicator(frame, confirmed):
     draw_indicator(frame, "angle", confirmed, row=3)

def draw_knee_indicator(frame, confirmed):
     draw_indicator(frame, "knee", confirmed, row=4)

ARM_ANGLE_MIN = 80
ARM_ANGLE_MAX = 95
ARM_ANGLE_TEST_MIN = 80  # once the reach is underway, the arm must not drop below this

def arm_angle_in_range(angles):
     if not angles:
          return False
     return any(ARM_ANGLE_MIN <= a <= ARM_ANGLE_MAX for a in angles.values())

def reach_side_from_angles(angles):
     """Which side ('right' preferred) is holding its arm angle in the 80-90 degree range,
     or None if neither is."""
     if not angles:
          return None
     for side in ("right", "left"):
          angle = angles.get(side)
          if angle is not None and ARM_ANGLE_MIN <= angle <= ARM_ANGLE_MAX:
               return side
     return None

def opposite_side(side):
     """The other side - used to check the knee opposite the reaching arm (the arm-side
     knee is typically occluded by the reaching arm itself in a side-on view)."""
     if side == "right":
          return "left"
     if side == "left":
          return "right"
     return None

def arm_angle_maintained(angles, side):
     """Whether the given side's arm angle has stayed at or above ARM_ANGLE_TEST_MIN during
     the active reach. No upper bound - only the arm dropping is a compensation."""
     if not angles or side is None:
          return True
     angle = angles.get(side)
     return angle is None or angle >= ARM_ANGLE_TEST_MIN

def wrist_x_px(camera, frame, side):
     """Current x pixel position of the given side's wrist, or None if not tracked/visible."""
     check = check_skeleton(camera)
     if not check or side is None:
          return None
     pose_landmarks = camera.landmarks[0]
     idx = R_WRIST_LANDMARK if side == "right" else L_WRIST_LANDMARK
     wrist = pose_landmarks[idx]
     if wrist.visibility < VISIBILITY_THRESHOLD:
          return None
     return wrist.x * frame.shape[1]

def reach_distance_cm(current_x_px, start_x_px, distance_cm, focal_length_px):
     """Horizontal wrist displacement converted to cm via the pinhole model, using the
     camera-to-subject distance captured at calibration as the depth reference."""
     if None in (current_x_px, start_x_px, distance_cm, focal_length_px):
          return None
     return abs(current_x_px - start_x_px) * distance_cm / focal_length_px

REACH_STILLNESS_TOLERANCE_CM = 3  # max wrist drift allowed during the hold, in real-world cm
REACH_HOLD_SECONDS = 3.0  # how long the wrist must stay still before the reach is locked in
KNEE_ANGLE_MIN_DEG = 160  # hip-knee-ankle vertex angle below this counts as a bent knee
KNEE_ANGLE_MAX_DEG = 180  # a straight leg reads ~180 degrees at the knee vertex
HIP_SHIFT_TOLERANCE_CM = 8  # max absolute hip shift from baseline before it counts as a footstep
TRUNK_ROTATION_TOLERANCE_RATIO = 0.15  # max fractional change in shoulder-to-shoulder width before it counts as trunk rotation

def wrist_still_px(current_x_px, anchor_x_px, distance_cm, focal_length_px):
     """Whether the wrist's x pixel position has stayed within REACH_STILLNESS_TOLERANCE_CM
     of the anchor, using the pinhole model to convert the cm tolerance to pixels."""
     if None in (current_x_px, anchor_x_px, distance_cm, focal_length_px):
          return False
     tolerance_px = REACH_STILLNESS_TOLERANCE_CM * focal_length_px / distance_cm
     return abs(current_x_px - anchor_x_px) <= tolerance_px

def check_legs(camera):
     check = check_skeleton(camera)
     if check:
          pose_landmarks = camera.landmarks[0]
          right_legs = all(
               pose_landmarks[i].visibility >= VISIBILITY_THRESHOLD
               for i in R_LEGS_LANDMARKS
          )
          left_legs = all(
               pose_landmarks[i].visibility >= VISIBILITY_THRESHOLD
               for i in L_LEGS_LANDMARKS
          )
          if right_legs or left_legs:
               print("I see legs")
          return right_legs or left_legs
     else:
          return False

def leg_points_px(camera, frame, side):
     """Current (hip_pt, knee_pt, ankle_pt) pixel points for the given side, or None if not tracked/visible."""
     check = check_skeleton(camera)
     if not check or side is None:
          return None
     pose_landmarks = camera.landmarks[0]
     hip_i = R_HIP_LANDMARKS if side == "right" else L_HIP_LANDMARKS
     knee_i = R_LEGS_LANDMARKS[1] if side == "right" else L_LEGS_LANDMARKS[1]
     ankle_i = R_LEGS_LANDMARKS[2] if side == "right" else L_LEGS_LANDMARKS[2]
     hip, knee, ankle = pose_landmarks[hip_i], pose_landmarks[knee_i], pose_landmarks[ankle_i]
     if min(hip.visibility, knee.visibility, ankle.visibility) < VISIBILITY_THRESHOLD:
          return None
     h, w = frame.shape[:2]
     return (hip.x * w, hip.y * h), (knee.x * w, knee.y * h), (ankle.x * w, ankle.y * h)

def draw_leg_lines(frame, points, color=(0, 255, 255)):
     hip_pt, knee_pt, ankle_pt = points
     cv2.line(frame, (int(hip_pt[0]), int(hip_pt[1])), (int(knee_pt[0]), int(knee_pt[1])), color, 10)
     cv2.line(frame, (int(knee_pt[0]), int(knee_pt[1])), (int(ankle_pt[0]), int(ankle_pt[1])), color, 10)

def knee_vertex_angle(points):
     """Angle (degrees) at the knee vertex between the hip and the ankle - i.e. the actual
     hip-knee-ankle knee flexion angle. ~180 degrees is a straight leg."""
     if points is None:
          return None
     hip_pt, knee_pt, ankle_pt = points
     return calculate_angle(hip_pt, knee_pt, ankle_pt)

def knee_straight(angle_deg):
     return angle_deg is None or KNEE_ANGLE_MIN_DEG <= angle_deg <= KNEE_ANGLE_MAX_DEG

def hip_shift_cm(static_points, current_points, distance_cm, focal_length_px):
     """Absolute horizontal shift (cm) of the hip itself relative to baseline. A large
     shift means the whole body translated - i.e. a footstep, not just a lean."""
     if None in (static_points, current_points, distance_cm, focal_length_px):
          return None
     static_hip_pt, _, _ = static_points
     current_hip_pt, _, _ = current_points
     return abs(current_hip_pt[0] - static_hip_pt[0]) * distance_cm / focal_length_px

def footstep_detected(hip_shift_cm_value):
     return hip_shift_cm_value is not None and hip_shift_cm_value > HIP_SHIFT_TOLERANCE_CM

def shoulder_width_px(camera, frame):
     """Current shoulder-to-shoulder pixel distance, or None if not tracked/visible."""
     check = check_skeleton(camera)
     if not check:
          return None
     pose_landmarks = camera.landmarks[0]
     l_shoulder = pose_landmarks[L_ARM_LANDMARKS[0]]
     r_shoulder = pose_landmarks[R_ARM_LANDMARKS[0]]
     if min(l_shoulder.visibility, r_shoulder.visibility) < VISIBILITY_THRESHOLD:
          return None
     w = frame.shape[1]
     return abs(l_shoulder.x - r_shoulder.x) * w

def trunk_rotated(current_width_px, baseline_width_px):
     """Whether the shoulder-to-shoulder width has changed enough from baseline to
     indicate the trunk rotated (turning foreshortens the shoulder line in view)."""
     if current_width_px is None or baseline_width_px is None or baseline_width_px == 0:
          return False
     return abs(current_width_px - baseline_width_px) / baseline_width_px > TRUNK_ROTATION_TOLERANCE_RATIO

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
     camera = Camera(camera=0, crop_w=720, crop_h=1280)
     skeleton = Skeleton(camera)

     state = "wait_distance"
     countdown_start = None
     static_points = None
     anchor_distance_cm = None
     captured_distance_cm = None
     reach_side = None
     reach_start_x_px = None
     reach_max_cm = None
     reach_still_anchor_x_px = None
     reach_still_start = None
     captured_reach_cm = None
     static_leg_points = None
     static_shoulder_width_px = None
     knee_ok = None
     final_leg_points = None

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
               if not window_sized:
                    fit_window_to_screen(window_name, frame)
                    window_sized = True

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

               if state == "wait_distance":
                    distance_cm = check_distance(skeleton, camera, frame)
                    if distance_cm is not None:
                         cv2.putText(
                              frame, f"distance: {distance_cm:.0f}cm", (10, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                         )
                    draw_message(frame, "stand still 100-200cm from camera")
                    if distance_in_range(distance_cm):
                         state = "countdown_distance"
                         countdown_start = time.time()
                         anchor_distance_cm = distance_cm
               elif state == "countdown_distance":
                    distance_cm = check_distance(skeleton, camera, frame)
                    if distance_cm is not None:
                         cv2.putText(
                              frame, f"distance: {distance_cm:.0f}cm", (10, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                         )
                    if not distance_in_range(distance_cm) or not distance_still(distance_cm, anchor_distance_cm):
                         state = "wait_distance"
                         anchor_distance_cm = None
                    else:
                         remaining = 3 - int(time.time() - countdown_start)
                         if remaining <= 0:
                              captured_distance_cm = distance_cm
                              state = "wait_arms"
                         else:
                              draw_message(frame, str(remaining))
               elif state == "wait_arms":
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
                              reach_side = reach_side_from_angles(arm_angles)
                              reach_start_x_px = wrist_x_px(camera, frame, reach_side)
                              static_leg_points = leg_points_px(camera, frame, opposite_side(reach_side))
                              static_shoulder_width_px = shoulder_width_px(camera, frame)
                              if reach_start_x_px is not None:
                                   print(f"Reach start: {reach_side} wrist at x={reach_start_x_px:.1f}px")
                         else:
                              draw_message(frame, str(remaining))
               elif state == "done":
                    current_leg_points = leg_points_px(camera, frame, opposite_side(reach_side))
                    current_shoulder_width_px = shoulder_width_px(camera, frame)

                    knee_angle = knee_vertex_angle(current_leg_points)
                    if knee_angle is not None:
                         print(f"knee angle: {knee_angle:.1f} deg")
                    hip_shift = hip_shift_cm(
                         static_leg_points, current_leg_points, captured_distance_cm, skeleton.focal_length_px,
                    )
                    knee_ok = knee_straight(knee_angle)
                    step_ok = not footstep_detected(hip_shift)
                    rotation_ok = not trunk_rotated(current_shoulder_width_px, static_shoulder_width_px)
                    arm_ok = arm_angle_maintained(arm_angles, reach_side)

                    if not arm_ok:
                         print(f"Reach invalidated (arm dropped below {ARM_ANGLE_TEST_MIN} deg); re-checking arm angle")
                         state = "wait_angle"
                         reach_max_cm = None
                         reach_still_anchor_x_px = None
                         reach_still_start = None
                         draw_message(frame, "arm dropped - hold arm at 90 degrees")
                    elif not (knee_ok and step_ok and rotation_ok):
                         reason = "knee bent" if not knee_ok else ("footstep" if not step_ok else "trunk rotation")
                         print(f"Reach invalidated ({reason}); repositioning")
                         state = "ready"
                         countdown_start = time.time()
                         reach_max_cm = None
                         reach_still_anchor_x_px = None
                         reach_still_start = None
                         draw_message(frame, f"{reason} detected - repositioning")
                    else:
                         if static_points is not None:
                              draw_lean_guideline(frame, *static_points, color=(0, 255, 255))
                         if lean_points is not None:
                              draw_lean_guideline(frame, *lean_points, color=(255, 0, 255))
                         if static_leg_points is not None:
                              draw_leg_lines(frame, static_leg_points, color=(255, 255, 0))
                         if current_leg_points is not None:
                              draw_leg_lines(frame, current_leg_points, color=(255, 0, 255))

                         current_wrist_x_px = wrist_x_px(camera, frame, reach_side)
                         reach_cm = reach_distance_cm(
                              current_wrist_x_px, reach_start_x_px, captured_distance_cm, skeleton.focal_length_px,
                         )
                         if reach_cm is not None:
                              if reach_max_cm is None or reach_cm > reach_max_cm:
                                   reach_max_cm = reach_cm
                              cv2.putText(
                                   frame, f"reach: {reach_cm:.1f}cm (max {reach_max_cm:.1f}cm)", (10, 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                              )

                              wrist_still = wrist_still_px(
                                   current_wrist_x_px, reach_still_anchor_x_px, captured_distance_cm, skeleton.focal_length_px,
                              )
                              if not wrist_still:
                                   reach_still_anchor_x_px = current_wrist_x_px
                                   reach_still_start = time.time()
                              else:
                                   hold_remaining = REACH_HOLD_SECONDS - (time.time() - reach_still_start)
                                   if hold_remaining <= 0:
                                        captured_reach_cm = reach_max_cm
                                        final_leg_points = current_leg_points
                                        print(f"Reach end: max reach = {captured_reach_cm / 2.54:.1f}in ({reach_side})")
                                        state = "finished"
                                   else:
                                        draw_message(frame, f"hold... {hold_remaining:.1f}")
                         else:
                              reach_still_anchor_x_px = None
               elif state == "ready":
                    if static_points is not None:
                         draw_lean_guideline(frame, *static_points, color=(0, 255, 255))
                    if static_leg_points is not None:
                         draw_leg_lines(frame, static_leg_points, color=(0, 255, 255))
                    remaining = 3 - int(time.time() - countdown_start)
                    if remaining <= 0:
                         state = "done"
                         static_points = lean_points
                         static_leg_points = leg_points_px(camera, frame, opposite_side(reach_side))
                         static_shoulder_width_px = shoulder_width_px(camera, frame)
                         reach_start_x_px = wrist_x_px(camera, frame, reach_side)
                    else:
                         draw_message(frame, f"reposition... {remaining}")
               elif state == "finished":
                    if static_points is not None:
                         draw_lean_guideline(frame, *static_points, color=(0, 255, 255))
                    if static_leg_points is not None:
                         draw_leg_lines(frame, static_leg_points, color=(128, 0, 128))
                    if final_leg_points is not None:
                         draw_leg_lines(frame, final_leg_points, color=(0, 0, 0))
                    cv2.putText(
                         frame, f"reach: {captured_reach_cm:.1f}cm (captured)", (10, 80),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    )

               distance_confirmed = state not in ("wait_distance", "countdown_distance")
               arms_confirmed = state not in ("wait_distance", "countdown_distance", "wait_arms", "countdown_arms")
               legs_confirmed = state not in (
                    "wait_distance", "countdown_distance", "wait_arms", "countdown_arms", "wait_legs", "countdown_legs",
               )
               angle_confirmed = state in ("done", "ready", "finished")
               knee_confirmed = state == "finished" or (state == "done" and bool(knee_ok))
               draw_distance_indicator(frame, distance_confirmed)
               draw_arms_indicator(frame, arms_confirmed)
               draw_legs_indicator(frame, legs_confirmed)
               draw_angle_indicator(frame, angle_confirmed)
               draw_knee_indicator(frame, knee_confirmed)

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
