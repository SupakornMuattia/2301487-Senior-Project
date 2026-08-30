from programs.checks.legs import check_skeleton, VISIBILITY_THRESHOLD, R_LEGS_LANDMARKS, L_LEGS_LANDMARKS
from programs.checks.arms import R_ARM_LANDMARKS, L_ARM_LANDMARKS

R_SHOULDER_LANDMARK = R_ARM_LANDMARKS[0]
L_SHOULDER_LANDMARK = L_ARM_LANDMARKS[0]
R_HIP_LANDMARK = R_LEGS_LANDMARKS[0]
L_HIP_LANDMARK = L_LEGS_LANDMARKS[0]

FACING_RATIO_TOLERANCE = 0.15  # max fractional drift from the baseline ratio before it counts as turned
Z_DIFF_TOLERANCE = 0.08  # max left/right depth gap (normalized, same scale as x) before it counts as turned
SIDEWAYS_Z_DIFF_MIN = 0.35  # min left/right depth gap before it counts as turned side-on

def shoulder_width_px(camera, frame):
     check = check_skeleton(camera)
     if not check:
          return None
     pose_landmarks = camera.landmarks[0]
     l_shoulder = pose_landmarks[L_SHOULDER_LANDMARK]
     r_shoulder = pose_landmarks[R_SHOULDER_LANDMARK]
     if min(l_shoulder.visibility, r_shoulder.visibility) < VISIBILITY_THRESHOLD:
          return None
     w = frame.shape[1]
     return abs(l_shoulder.x - r_shoulder.x) * w

def hip_width_px(camera, frame):
     check = check_skeleton(camera)
     if not check:
          return None
     pose_landmarks = camera.landmarks[0]
     l_hip = pose_landmarks[L_HIP_LANDMARK]
     r_hip = pose_landmarks[R_HIP_LANDMARK]
     if min(l_hip.visibility, r_hip.visibility) < VISIBILITY_THRESHOLD:
          return None
     w = frame.shape[1]
     return abs(l_hip.x - r_hip.x) * w

def shoulder_hip_ratio(camera, frame):
     """Current shoulder-width-to-hip-width ratio, or None if either isn't tracked/visible."""
     shoulder_width = shoulder_width_px(camera, frame)
     hip_width = hip_width_px(camera, frame)
     if shoulder_width is None or hip_width is None or hip_width == 0:
          return None
     return shoulder_width / hip_width

def facing_camera(current_ratio, baseline_ratio, tolerance=FACING_RATIO_TOLERANCE):
     """Whether the current shoulder/hip ratio has stayed within `tolerance` (fractional)
     of the baseline ratio captured while the user was confirmed facing the camera."""
     if current_ratio is None or baseline_ratio in (None, 0):
          return False
     return abs(current_ratio - baseline_ratio) / baseline_ratio <= tolerance

def shoulder_z_diff(camera):
     """Depth gap between left/right shoulders, or None if either isn't tracked/visible."""
     check = check_skeleton(camera)
     if not check:
          return None
     pose_landmarks = camera.landmarks[0]
     l_shoulder = pose_landmarks[L_SHOULDER_LANDMARK]
     r_shoulder = pose_landmarks[R_SHOULDER_LANDMARK]
     if min(l_shoulder.visibility, r_shoulder.visibility) < VISIBILITY_THRESHOLD:
          return None
     return abs(l_shoulder.z - r_shoulder.z)

def hip_z_diff(camera):
     """Depth gap between left/right hips, or None if either isn't tracked/visible."""
     check = check_skeleton(camera)
     if not check:
          return None
     pose_landmarks = camera.landmarks[0]
     l_hip = pose_landmarks[L_HIP_LANDMARK]
     r_hip = pose_landmarks[R_HIP_LANDMARK]
     if min(l_hip.visibility, r_hip.visibility) < VISIBILITY_THRESHOLD:
          return None
     return abs(l_hip.z - r_hip.z)

def facing_camera_live(camera, tolerance=Z_DIFF_TOLERANCE):
     """Whether the user is currently facing the camera, judged frame-by-frame
     from left/right shoulder+hip depth symmetry. No calibration baseline
     needed: a turned torso pushes one side's landmarks closer to the camera
     than the other, widening shoulder_z_diff/hip_z_diff past `tolerance`."""
     s_diff = shoulder_z_diff(camera)
     h_diff = hip_z_diff(camera)
     if s_diff is None or h_diff is None:
          return False
     return s_diff <= tolerance and h_diff <= tolerance

def facing_sideways_live(camera, tolerance=SIDEWAYS_Z_DIFF_MIN):
     """Whether the user is currently turned side-on to the camera, judged
     frame-by-frame from left/right shoulder+hip depth asymmetry. A profile
     stance pushes the near side's shoulder/hip well in front of the far
     side, widening shoulder_z_diff/hip_z_diff past `tolerance`."""
     s_diff = shoulder_z_diff(camera)
     h_diff = hip_z_diff(camera)
     if s_diff is None or h_diff is None:
          return False
     return s_diff >= tolerance and h_diff >= tolerance
