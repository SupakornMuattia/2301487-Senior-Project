# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository currently contains only planning documents (`README.md`, `Project Proposal.md`) — no source code, build system, or dependencies have been added yet. There are no build, lint, or test commands to run. Update this file once the codebase is scaffolded.

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
