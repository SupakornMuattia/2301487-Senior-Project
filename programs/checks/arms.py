R_ARM_LANDMARKS = [12, 14, 16]  # shoulder, elbow, wrist
L_ARM_LANDMARKS = [11, 13, 15]  # shoulder, elbow, wrist
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
          return right_arms or left_arms
     else:
          return False
