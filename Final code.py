# === Final FLIR Camera Calibration Code for DMS Project ===

import numpy as np
import cv2
import glob
import os

# =============== HARDWARE-SPECIFIC SETTINGS ===============
CHECKERBOARD = (10, 7)  # 11x8 squares = 10x7 inner corners
SQUARE_SIZE_MM = 25     # Size of checkerboard square in mm
FLIR_RESOLUTION = (1440, 1080)  # FLIR Firefly FFY-U3-16S2C-S resolution
IMAGE_EXT = 'jpg'  # Image format

# Sensor specs (Sony IMX296, 1/2.9")
SENSOR_WIDTH_MM = 6.2
SENSOR_HEIGHT_MM = 4.65
PIXEL_PITCH_MM = SENSOR_WIDTH_MM / FLIR_RESOLUTION[0]  # ≈ 0.0043 mm/px

# =============== INITIAL CAMERA MATRIX ===============
camera_matrix_init = np.array([
    [800, 0, FLIR_RESOLUTION[0]/2],
    [0, 800, FLIR_RESOLUTION[1]/2],
    [0, 0, 1]
], dtype=np.float32)

# Reasonable starting distortion (based on lens specs)
INITIAL_DIST_COEFFS = np.array([-0.3, 0.1, 0.01, 0.01, -0.01], dtype=np.float32)


def calibrate_flir():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = SQUARE_SIZE_MM * np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    images = glob.glob(f'calibration_images/*.{IMAGE_EXT}')
    if not images:
        print(f"❌ ERROR: No .{IMAGE_EXT} images found in 'calibration_images/'")
        return

    print(f"📸 Processing {len(images)} images...")
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"⚠️ WARNING: Could not read {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001))
            obj_points.append(objp)
            img_points.append(corners_refined)

            cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
            cv2.imshow(f'Image {i+1}', img)
            cv2.waitKey(200)

    cv2.destroyAllWindows()

    if len(obj_points) < 5:
        print(f"❌ ERROR: Only {len(obj_points)} valid calibration images. At least 5 required.")
        return

    print("\n📐 Starting camera calibration...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=obj_points,
        imagePoints=img_points,
        imageSize=FLIR_RESOLUTION,
        cameraMatrix=camera_matrix_init,
        distCoeffs=INITIAL_DIST_COEFFS,
        flags=(cv2.CALIB_USE_INTRINSIC_GUESS |
               cv2.CALIB_RATIONAL_MODEL |
               cv2.CALIB_FIX_ASPECT_RATIO)
    )

    # Clip distortion to reasonable range
    dist = np.clip(dist, -1.0, 1.0)

    # Calculate physical focal length
    focal_length_px = K[0, 0]
    focal_length_mm = focal_length_px * PIXEL_PITCH_MM

    print("\n=== FINAL CALIBRATION RESULTS ===")
    print(f"Reprojection Error: {ret:.3f} px (target <0.5)")
    print(f"Camera Matrix (K):\n{K}")
    print(f"Distortion Coefficients (k1, k2, p1, p2, k3, ...):\n{dist.ravel()}")
    print(f"Calculated Focal Length: {focal_length_mm:.2f} mm (expected ~8 mm)")

    os.makedirs('calibration_results', exist_ok=True)
    np.savez('calibration_results/flir_calibration_final.npz',
             K=K, dist=dist, resolution=FLIR_RESOLUTION,
             rvecs=rvecs, tvecs=tvecs,
             pixel_pitch_mm=PIXEL_PITCH_MM)

    if images:
        img = cv2.imread(images[0])
        h, w = img.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1)
        undistorted = cv2.undistort(img, K, dist, None, new_K)
        cv2.imwrite('calibration_results/comparison_final.jpg', np.hstack((img, undistorted)))
        print("\n✅ Undistorted example saved to 'calibration_results/comparison_final.jpg'")


if __name__ == "__main__":
    calibrate_flir()
