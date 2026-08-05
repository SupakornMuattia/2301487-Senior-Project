"""Open a webcam feed and center-crop it to a portrait, mobile-like frame, as a stand-in
for a phone camera before this prototype gets ported to Kotlin/CameraX on Android.

Webcams only support fixed landscape resolutions and can't rotate their sensor, so this
always captures at the camera's native max resolution and then center-crops a portrait
region out of it in software — the subject stays upright (no rotation), only the frame
edges are cut off. --w/--h set the crop size; cropping can only shrink a frame, so --h
can be at most the camera's native frame height (e.g. <=720).

Usage:
    python camera_preview.py [--w 405] [--h 720] [--camera 0]

Press 'q' or Esc to quit.
"""

import argparse
import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w", type=int, default=405, help="Desired crop width")
    parser.add_argument("--h", type=int, default=720, help="Desired crop height")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    return parser.parse_args()


def center_crop(frame, target_w: int, target_h: int):
    h, w = frame.shape[:2]
    crop_w = min(target_w, w)
    crop_h = min(target_h, h)
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return frame[y0 : y0 + crop_h, x0 : x0 + crop_w]


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    # Ask for more than any webcam supports so the driver snaps to its native max
    # resolution, rather than snapping down toward the small portrait crop size.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = min(args.w, actual_width)
    crop_h = min(args.h, actual_height)
    print(f"Camera captures {actual_width}x{actual_height}, cropping to {crop_w}x{crop_h}")

    window_name = "Camera Preview (q or Esc to quit)"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera")
                break

            frame = cv2.flip(frame, 1)
            frame = center_crop(frame, args.w, args.h)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 = Esc
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
