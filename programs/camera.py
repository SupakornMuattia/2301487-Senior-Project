import cv2


class Camera:
    def __init__(self, camera: int = 0, crop_w: int = 405, crop_h: int = 720):
        self.camera = camera
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.cap = None
        self.landmarker = None

    @staticmethod
    def frame_crop(frame, target_w: int, target_h: int):
        h, w = frame.shape[:2]
        crop_w = min(target_w, w)
        crop_h = min(target_h, h)
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        return frame[y0 : y0 + crop_h, x0 : x0 + crop_w]

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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

    def run_preview(self):
        self.open_camera()
        window_name = "Camera Preview (q or Esc to quit)"
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Failed to read frame from camera")
                    break

                frame = cv2.flip(frame, 1)
                frame = self.frame_crop(frame, self.crop_w, self.crop_h)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 27 = Esc
                    break
        finally:
            self.release()
