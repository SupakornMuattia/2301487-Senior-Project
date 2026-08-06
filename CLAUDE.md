# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Early scaffolding stage. Alongside the planning documents (`README.md`, `Project Proposal.md`), the repo now has an initial Python prototype for the camera pipeline:

- `camera_preview.py` — standalone script prototype: opens a webcam via OpenCV, center-crops the feed to a portrait "phone-like" frame (stand-in for a real phone camera before this is ported to Kotlin/CameraX on Android).
- `main/camera.py` — the same camera-preview logic refactored into a `Camera` class (`open`/`run_preview`/`release`, plus static `parse_args`/`center_crop` helpers). This is the version to build on going forward; `camera_preview.py` is the earlier functional-style draft it was refactored from.
- `main/activity/`, `main/gait/`, `main/leaning/` — empty directories, presumably placeholders for upcoming TUG-related analysis modules (activity detection, gait analysis, postural leaning). Not yet implemented.
- `requirements.txt` — currently just `opencv-python` and `numpy`. MediaPipe and Matplotlib from the proposed architecture are not yet added.
- `venv/` — local virtualenv, gitignored (along with `__pycache__/` and `*.pyc`).

No Flutter/Dart mobile shell exists yet — everything so far is the Python vision-side prototype, run directly (`python main/camera.py` or `python camera_preview.py`), not yet wired into any app or pipeline. There is no build, lint, or test command/framework configured. Update this section as the pose-estimation (MediaPipe) and TUG-metric logic get added under `main/`.

## Project overview

A mobile application for detecting and measuring balance impairment in elderly users, based on the clinical **Timed Up and Go (TUG)** test. The app is intended to let users self-administer a TUG-style balance assessment using a smartphone's camera and sensors, replacing manual stopwatch/observer-based scoring with automated pose tracking.

Per the proposal, the planned architecture is:

- **Mobile UI**: Flutter (Dart) — Android Studio as the primary IDE.
- **Pose estimation / vision processing**: Python, using MediaPipe's Pose Landmark solution (real-time body landmark detection) and OpenCV for image/frame processing.
- **Sensor fusion / motion tracking**: ARCore, plus device accelerometer and gyroscope, for tracking device/body movement during the test.
- **Numerical processing**: NumPy for landmark coordinate computation and TUG timing/metric calculations.
- **Results visualization**: Matplotlib for plotting assessment results.
- **Camera pipeline**: raw camera image stream (YUV420) feeds into the pose-processing pipeline.

Because the vision/ML pipeline (Python + MediaPipe/OpenCV) and the mobile client (Flutter/Dart) are different languages, expect this project to eventually split into at least two components — a pose-processing/analysis piece and a Flutter UI/app shell — that need to communicate (e.g. via on-device inference bindings, a backend service, or bundled native modules). No such split exists in the repo yet.
