# Methodology

This document explains the underlying computer vision and math behind the
calibration pipeline in this repo.

## 1. The pinhole camera model

A camera maps 3D points in the world onto a 2D image plane. Under the ideal
pinhole model, this mapping is described by:

```
[u v 1]^T ≈ K [R | t] [X Y Z 1]^T
```

where:
- **K** (3x3) is the **intrinsic matrix** — focal length (`fx`, `fy`),
  principal point (`cx`, `cy`), and skew (assumed 0 for modern sensors).
- **[R | t]** are the **extrinsic parameters** — the rotation and translation
  that relate the camera's coordinate frame to the world/object frame.
- Real lenses also introduce **distortion**, which is not part of the linear
  pinhole model and must be corrected separately (Section 3).

```
K = | fx   0   cx |
    |  0  fy   cy |
    |  0   0    1 |
```

## 2. Estimating K and the extrinsics: Zhang's method

This toolkit uses `cv2.calibrateCamera`, an implementation of **Zhang's
flexible calibration method** (Zhang, 2000). The approach:

1. Present the camera with a **planar checkerboard pattern** of known
   geometry (real-world square size), photographed from several different
   angles and distances.
2. Detect the checkerboard's inner corners in each image
   (`cv2.findChessboardCorners`), refined to sub-pixel accuracy
   (`cv2.cornerSubPix`).
3. Because the 3D positions of the checkerboard corners *relative to each
   other* are known exactly (they're a regular grid), each image gives a set
   of 2D-3D correspondences.
4. With enough correspondences across enough distinct viewpoints, the system
   of equations becomes solvable via a homography-based closed-form initial
   estimate, followed by **non-linear refinement (Levenberg-Marquardt)** that
   jointly optimizes intrinsics, extrinsics, and distortion coefficients to
   minimize reprojection error.

## 3. Lens distortion

Real lenses deviate from the ideal pinhole model in two ways:

**Radial distortion** — caused by the spherical shape of the lens; produces
barrel distortion (image edges bow outward) or pincushion distortion (edges
bow inward). Modeled with coefficients `k1, k2, k3`:

```
x_corrected = x(1 + k1*r^2 + k2*r^4 + k3*r^6)
y_corrected = y(1 + k1*r^2 + k2*r^4 + k3*r^6)
```

**Tangential distortion** — caused by imperfect lens-sensor alignment.
Modeled with `p1, p2`:

```
x_corrected = x + [2*p1*x*y + p2*(r^2 + 2*x^2)]
y_corrected = y + [p1*(r^2 + 2*y^2) + 2*p2*x*y]
```

Once these five coefficients are known, `cv2.undistort` (or an explicit
pixel remapping via `cv2.initUndistortRectifyMap`, used in
`visualize.py`) can correct any image from that camera.

## 4. Reprojection error: measuring calibration quality

Once K, distortion coefficients, and per-image extrinsics are estimated, the
known 3D checkerboard points are **projected back** into each image using the
recovered camera model. The Euclidean distance (in pixels) between these
projected points and the originally detected corner positions is the
**reprojection error**.

```
error_i = || detected_corner_i - project(3D_point_i, K, dist, R, t) ||
```

A lower mean reprojection error means the recovered camera model more
accurately explains the observed data. In this toolkit, results are
reported per-image and as a mean, matching common industry practice of
tracking a target threshold (commonly ~0.5 px for demanding applications
such as driver monitoring, AR registration, or precision robotics).

## 5. Why this matters for downstream systems

Any system that relies on the camera to make geometric measurements —
distance estimation, gaze/pose estimation, AR overlay alignment, feature
triangulation — inherits the calibration's accuracy as a hard floor.
An uncalibrated or poorly calibrated camera introduces systematic geometric
error that no downstream algorithm can fully undo. This is why calibration
is treated as a first-class, independently validated step rather than an
implementation detail.

## References

- Z. Zhang, "A flexible new technique for camera calibration," *IEEE
  Transactions on Pattern Analysis and Machine Intelligence*, vol. 22,
  no. 11, pp. 1330-1334, 2000.
- OpenCV documentation: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
