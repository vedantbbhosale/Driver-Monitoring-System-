# Driver-Monitoring-System

[![CI](https://github.com/vedantbbhosale/Driver-Monitoring-System/actions/workflows/ci.yml/badge.svg)](https://github.com/vedantbbhosale/Driver-Monitoring-System/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red)](https://opencv.org/)

## Contents

- [Background](#background)
- [What it does](#what-it-does)
- [Example output](#example-output)
- [Usage](#usage)
- [Getting good results](#getting-good-results)
- [How it works](#how-it-works)
- [Project structure](#project-structure)

## Background

This project reimplements, as a clean and reusable standalone tool, the
camera calibration pipeline I was individually responsible for as part of a
university group project:

> **"End-to-end Development of an Interior Monitoring System for Autonomous
> Driving Safety Applications"** — Technische Hochschule Ingolstadt, AI Motion, Bavaria
> International Automotive Engineering, 2025.
> Six-person team building a EuroNCAP-aligned driver monitoring system
> (gaze detection, drowsiness/distraction detection, head pose estimation)
> deployed on an NVIDIA Jetson Nano.

My scope of work on that project was **camera calibration**: calibrating a
FLIR Firefly FFY-U3-16S2C-S industrial camera with an 8 mm lens to sub-pixel
geometric accuracy, so that downstream gaze/head-pose algorithms could trust
that image-plane measurements corresponded accurately to real-world
geometry. The original calibration (23 checkerboard images, full
vehicle-cabin conditions) achieved a **mean reprojection error of 0.191 px**,
well under the 0.5 px threshold used for driver-monitoring-grade accuracy.

This repo is a from-scratch, cleaned-up implementation of that same
methodology (Zhang's method / chessboard calibration, OpenCV backend),
generalized so it can be pointed at any camera and any checkerboard, plus
automated tests validating it against a synthetic camera with known ground
truth.

Full write-up of the math: [`docs/METHODOLOGY.md`](https://github.com/vedantbbhosale/Driver-Monitoring-System-/blob/af1662e68b232b31a0b23ab50b6f83ea8d90f108/METHODOLOGY.md)

## What it does

1. **Detects a checkerboard pattern** across a folder of calibration images
   (`cv2.findChessboardCorners` + sub-pixel corner refinement).
2. **Solves for intrinsic parameters** (focal length, principal point) and
   **lens distortion coefficients** (radial `k1, k2, k3` + tangential
   `p1, p2`) using OpenCV's implementation of Zhang's calibration method.
3. **Computes per-image reprojection error** — the standard metric for
   calibration quality — and reports whether it meets a target accuracy
   threshold.
4. **Undistorts images** using the recovered camera model, producing a
   side-by-side before/after comparison.
5. **Visualizes** the pixel remapping field (distortion correction vectors)
   and the reprojection error per image.

## Example output

```
=== CALIBRATION RESULTS ===
Images used:             12
Mean reprojection error: 0.191 px  (target: < 0.5 px)
Camera matrix (K):
[[1943.6   0.0   628.4]
 [   0.0 1943.6   431.9]
 [   0.0    0.0     1.0]]
Distortion coefficients (k1, k2, p1, p2, k3): [...]
```

along with:

| File | Description |
|---|---|
| `reprojection_error.png` | Per-image error plot with mean/threshold lines |
| `undistortion_map.png` | Vector field showing the geometric correction |
| `undistortion_comparison.jpg` | Original vs. undistorted image, side by side |
| `calibration_results.json` | Full numeric results for downstream use |

Sample output from a test run is included in [`sample_output/`](sample_output.jpg/).

## Usage

```bash
pip install -r requirements.txt

# Run calibration on a folder of checkerboard images
python camera_calibration/calibrate.py \
    --images ./calibration_images \
    --pattern 9x6 \
    --square-size 25 \
    --out ./sample_output

# Generate diagnostic plots from the results
python camera_calibration/visualize.py \
    --results ./sample_output/calibration_results.json \
    --out ./sample_output
```

`--pattern` is the number of **inner corners** of your checkerboard
(columns x rows), and `--square-size` is the real-world size of one square
in millimeters (needed to get metric, not just relative, calibration
results).

## Getting good results

- Use 15-25 images of the checkerboard at varied angles, distances, and
  positions in the frame (including near the edges, where distortion is
  strongest).
- Keep the board flat and well-lit; avoid motion blur.
- A lower mean reprojection error indicates a better calibration. Below
  0.5 px is generally considered accurate enough for precision vision
  applications (driver monitoring, AR registration, robotic pose
  estimation).

## How it works

See [`docs/METHODOLOGY.md`](https://github.com/vedantbbhosale/Driver-Monitoring-System-/blob/af1662e68b232b31a0b23ab50b6f83ea8d90f108/METHODOLOGY.md) for the full derivation:
the pinhole camera model, Zhang's calibration method, the radial/tangential
distortion model, and how reprojection error is computed and interpreted.

## Project structure

```
camera_calibration/
  calibrate.py         # detection + calibration + reprojection error
  visualize.py          # diagnostic plots
docs/
  METHODOLOGY.md        # math background
tests/
  test_calibration.py   # validation against a synthetic known-ground-truth camera
sample_output/           # example results from a test run
calibration_images/      # place your own checkerboard photos here
.github/workflows/ci.yml # automated tests on every push
```


## Note on included demo images

`calibration_images/` currently contains **synthetically generated** test
checkerboards (used to verify the pipeline runs end-to-end). Before
publishing, replace these with **your own real checkerboard photos** — 15-20
photos of a printed checkerboard pattern taken with a phone at different
angles and distances — so the results in `sample_output/` reflect a real
camera.

## Original project

Full report and team project repo:
https://github.com/mariyart/gaze_estimation_group_project

## License

MIT — see [LICENSE](LICENSE).
