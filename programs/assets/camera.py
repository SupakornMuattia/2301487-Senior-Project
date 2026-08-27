import ctypes

import cv2

url = "https://192.168.1.28:8080/video"
url_2 = "https://10.99.27.126:8080/video"


class Camera:
    def __init__(self, camera: int = 0, crop_w: int = 720, crop_h: int = 1280):
        self.camera = camera
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.cap = None
        self.landmarks = None
        self.face_landmarks = None

    @staticmethod
    def frame_crop(frame, target_w: int, target_h: int):
        h, w = frame.shape[:2]
        crop_w = min(target_w, w)
        crop_h = min(target_h, h)
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        return frame[y0 : y0 + crop_h, x0 : x0 + crop_w]

    def open_camera(self):
        self.cap = cv2.VideoCapture(url)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera}")

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        crop_w = min(self.crop_w, actual_width)
        crop_h = min(self.crop_h, actual_height)
        print(f"Camera captures {actual_width}x{actual_height}, cropping to {crop_w}x{crop_h}")

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

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
        self.open_camera()
        window_name = "Camera Preview (q or Esc to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_sized = False
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Failed to read frame from camera")
                    break

                frame = cv2.flip(frame, 1)
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                # frame = self.frame_crop(frame, self.crop_w, self.crop_h)

                if not window_sized:
                    self.fit_window_to_screen(window_name, frame)
                    window_sized = True

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 27 = Esc
                    break
        finally:
            self.release()

if __name__ == "__main__":
    camera = Camera()
    camera.run_preview()