R_LEGS_LANDMARKS = [24, 26, 28]  # hip, knee, ankle
L_LEGS_LANDMARKS = [23, 25, 27]  # hip, knee, ankle
VISIBILITY_THRESHOLD = 0.6

def check_skeleton(camera):
     if camera.landmarks:
          return True
     else:
          return False

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
          return right_legs or left_legs
     else:
          return False
