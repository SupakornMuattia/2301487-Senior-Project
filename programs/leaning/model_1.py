import cv2

from programs.skeleton import Skeleton
from programs.camera import Camera


ARM_LANDMARKS = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
VISIBILITY_THRESHOLD = 0.5

def check_skeleton(camera):
     if camera.landmarker:
          return True
     else:
          return False

def check_arms(camera):
     check = check_skeleton(camera)
     if check:
          pose_landmarks = camera.landmarker[0]
          have_arms = all(
               pose_landmarks[i].visibility >= VISIBILITY_THRESHOLD
               for i in ARM_LANDMARKS
          )
          if have_arms:
               print("I see arms")
          else:
               print("I don't see arms")
          return have_arms
     else:
          print("does not found skeleton")
          return False
     
def check_legs():
     print("I see legs")
     return

if __name__ == "__main__":
     camera = Camera(camera=0, crop_w=405, crop_h=720)
     skeleton = Skeleton(camera)

     camera.open_camera()
     try:
          while True:
               ok, frame = camera.cap.read()
               if not ok:
                    print("Failed to read frame from camera")
                    break

               frame = cv2.flip(frame, 1)
               frame = Camera.frame_crop(frame, camera.crop_w, camera.crop_h)

               result = skeleton.detect(frame)
               if result is not None and result.pose_landmarks:
                    frame = skeleton.draw_landmarks(frame, result)

               check_arms(camera)

               cv2.imshow("Frame", frame)
               key = cv2.waitKey(1) & 0xFF
               if key in (ord("q"), 27):  # 27 = Esc
                    break
               elif key == ord("t"):  # toggle skeleton detection
                    skeleton.disable() if skeleton.enabled else skeleton.enable()
     finally:
          skeleton.landmarker.close()
          camera.release()
