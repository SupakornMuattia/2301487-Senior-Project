import re
import subprocess


class Calibration:
    """Reads Camera2 CameraCharacteristics off a phone via `adb shell dumpsys
    media.camera` and derives focal_length_px (pinhole model) from them."""
    adb = r"D:\Senior Project\2301487-Senior-Project\platform-tools\adb.exe"
    def __init__(
        self,
        facing: str = "back",
        camera_id: str = None,
        device_serial: str = None,
        adb_path: str = adb,
    ):
        self.facing = facing
        self.camera_id = camera_id
        self.device_serial = device_serial
        self.adb_path = adb_path

    def _dumpsys_camera(self) -> str:
        cmd = [self.adb_path]
        if self.device_serial:
            cmd += ["-s", self.device_serial]
        cmd += ["shell", "dumpsys", "media.camera"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            raise RuntimeError("adb not found on PATH; install platform-tools or pass adb_path")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"adb dumpsys media.camera failed: {e.stderr.strip()}")
        return result.stdout

    @staticmethod
    def _camera_block(dump: str, camera_id: str) -> str:
        """Isolate the static-characteristics section for camera_id, delimited
        by Android's "== Camera HAL device device@.../internal/<id> (vX.X)
        static information: ==" ... "... dumpState: ==" markers. Falls back to
        the whole dump if those markers aren't present (older/other vendors)."""
        id_re = re.escape(camera_id)
        start_match = re.search(
            rf"== Camera HAL device device@[\d.]+/internal/{id_re} \(v[\d.]+\) static information: ==",
            dump,
        )
        if not start_match:
            return dump
        start = start_match.end()
        end_match = re.search(
            rf"== Camera HAL device device@[\d.]+/internal/{id_re} \(v[\d.]+\) dumpState: ==",
            dump[start:],
        )
        end = start + end_match.start() if end_match else len(dump)
        return dump[start:end]

    def _resolve_camera_id(self, dump: str) -> str:
        """self.camera_id if set explicitly, otherwise the id of the first
        camera whose "Facing:" line matches self.facing ("back"/"front")."""
        if self.camera_id is not None:
            return self.camera_id

        facing_word = {"back": "Back", "front": "Front"}.get(self.facing.lower())
        if facing_word is None:
            raise ValueError(f"facing must be 'back' or 'front', got {self.facing!r}")

        for match in re.finditer(
            r"== Camera HAL device device@[\d.]+/internal/(\d+) \(v[\d.]+\) static information: ==",
            dump,
        ):
            candidate_id = match.group(1)
            block = self._camera_block(dump, candidate_id)
            if re.search(rf"Facing:\s*{facing_word}\b", block):
                return candidate_id

        raise RuntimeError(f"No {self.facing}-facing camera found in dumpsys media.camera output")

    @staticmethod
    def _parse_focal_length_mm(block: str) -> float:
        match = re.search(
            r"android\.lens\.info\.availableFocalLengths[^\n]*\n\s*\[\s*([\d.]+)", block
        )
        if not match:
            raise ValueError("android.lens.info.availableFocalLengths not found in dumpsys output")
        return float(match.group(1))

    @staticmethod
    def _parse_sensor_physical_size_mm(block: str) -> tuple:
        match = re.search(
            r"android\.sensor\.info\.physicalSize[^\n]*\n\s*\[\s*([\d.]+)\s+([\d.]+)", block
        )
        if not match:
            raise ValueError("android.sensor.info.physicalSize not found in dumpsys output")
        return float(match.group(1)), float(match.group(2))

    @staticmethod
    def _parse_pixel_array_size(block: str) -> tuple:
        match = re.search(
            r"android\.sensor\.info\.pixelArraySize[^\n]*\n\s*\[\s*(\d+)\s+(\d+)", block
        )
        if not match:
            raise ValueError("android.sensor.info.pixelArraySize not found in dumpsys output")
        return int(match.group(1)), int(match.group(2))

    def focal_length_px(self, image_width_px: int = None) -> float:
        """focal_length_px = focal_length_mm * (image_width_px / sensor_width_mm).

        image_width_px should be the width of the frames actually being
        processed (e.g. Camera.crop_w / the captured stream width), which is
        not necessarily the sensor's full native pixel array width. Defaults
        to the native pixel array width reported by Camera2 if omitted.
        """
        dump = self._dumpsys_camera()
        camera_id = self._resolve_camera_id(dump)
        block = self._camera_block(dump, camera_id)

        focal_length_mm = self._parse_focal_length_mm(block)
        sensor_width_mm, _ = self._parse_sensor_physical_size_mm(block)

        if image_width_px is None:
            image_width_px, _ = self._parse_pixel_array_size(block)

        return focal_length_mm * (image_width_px / sensor_width_mm)


if __name__ == "__main__":
    calibration = Calibration(facing="front")
    print(f"focal_length_px = {calibration.focal_length_px(1280):.2f}")
