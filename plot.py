# === Final FLIR Camera Calibration Code for DMS Project with Error Visualization ===

import numpy as np
import cv2
import glob
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt

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
    image_paths = []  # Store image paths for later reference

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
            
        image_paths.append(fname)  # Store valid image path
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001))
            obj_points.append(objp)
            img_points.append(corners_refined)

            # Draw and display detected corners
            debug_img = img.copy()
            cv2.drawChessboardCorners(debug_img, CHECKERBOARD, corners_refined, ret)
            cv2.imshow(f'Image {i+1}/{len(images)}', debug_img)
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

    # =============== REPROJECTION ERROR ANALYSIS ===============
    print("\n🔍 Calculating per-image reprojection errors...")
    per_image_errors = []
    all_errors = []
    
    # Calculate errors for each image
    for i in range(len(obj_points)):
        img_points_repro, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        img_points_repro = img_points_repro.reshape(-1, 2)
        errors = np.linalg.norm(img_points[i] - img_points_repro, axis=1)
        per_image_errors.append(np.mean(errors))
        all_errors.extend(errors)
    
    # Create results directory
    os.makedirs('calibration_results', exist_ok=True)
    
    # =============== ERROR VISUALIZATION ===============
    # 1. Reprojection Error Plot
    plt.figure(figsize=(12, 6))
    indices = range(len(per_image_errors))
    bars = plt.bar(indices, per_image_errors, color='skyblue')
    
    # Highlight high-error images
    for idx, err in enumerate(per_image_errors):
        if err > 0.5:  # Euro NCAP threshold
            bars[idx].set_color('salmon')
    
    # Add reference lines
    plt.axhline(y=ret, color='blue', linestyle='--', label=f'Overall RMS: {ret:.3f} px')
    plt.axhline(y=0.5, color='red', linestyle='-', label='Euro NCAP Threshold (0.5 px)')
    
    # Formatting
    plt.xlabel('Calibration Image Index')
    plt.ylabel('Mean Reprojection Error (px)')
    plt.title('Reprojection Error per Calibration Image')
    plt.xticks(indices)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    error_plot_path = os.path.join('calibration_results', 'reprojection_errors.png')
    plt.savefig(error_plot_path, dpi=150)
    plt.close()
    print(f"✅ Reprojection error plot saved to {error_plot_path}")

    # 2. Error Distribution Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=30, color='teal', alpha=0.7)
    plt.axvline(x=np.mean(all_errors), color='red', linestyle='dashed', 
                linewidth=1.5, label=f'Mean: {np.mean(all_errors):.3f} px')
    plt.xlabel('Reprojection Error (px)')
    plt.ylabel('Number of Points')
    plt.title('Reprojection Error Distribution Across All Points')
    plt.legend()
    plt.grid(alpha=0.2)
    hist_path = os.path.join('calibration_results', 'error_distribution.png')
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"✅ Error distribution histogram saved to {hist_path}")

    # 3. Visualize worst-case image
    worst_idx = np.argmax(per_image_errors)
    worst_img = cv2.imread(image_paths[worst_idx])
    worst_img_points_repro, _ = cv2.projectPoints(
        obj_points[worst_idx], rvecs[worst_idx], tvecs[worst_idx], K, dist
    )
    
    # Draw detected vs reprojected points
    debug_img = worst_img.copy()
    for j, (detected, reprojected) in enumerate(zip(img_points[worst_idx], worst_img_points_repro)):
        color = (0, 255, 0)  # Green for good matches
        error = np.linalg.norm(detected - reprojected)
        if error > 1.0:  # Highlight large errors in red
            color = (0, 0, 255)
            cv2.putText(debug_img, f'{error:.2f}', 
                        tuple(reprojected.astype(int).ravel()), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
        cv2.circle(debug_img, tuple(detected.astype(int).ravel()), 4, (0, 255, 255), -1)  # Yellow: detected
        cv2.circle(debug_img, tuple(reprojected.astype(int).ravel()), 3, color, -1)        # Green/Red: reprojected
        cv2.line(debug_img, 
                 tuple(detected.astype(int).ravel()), 
                 tuple(reprojected.astype(int).ravel()), 
                 (255, 255, 255), 1)
    
    # Add info overlay
    cv2.putText(debug_img, f'Image {worst_idx+1}: Mean Error = {per_image_errors[worst_idx]:.3f} px', 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    worst_case_path = os.path.join('calibration_results', 'worst_case_analysis.jpg')
    cv2.imwrite(worst_case_path, debug_img)
    print(f"✅ Worst-case analysis saved to {worst_case_path}")

    # =============== SAVE RESULTS ===============
    print("\n=== FINAL CALIBRATION RESULTS ===")
    print(f"Reprojection Error: {ret:.3f} px (target <0.5)")
    print(f"Camera Matrix (K):\n{K}")
    print(f"Distortion Coefficients (k1, k2, p1, p2, k3, ...):\n{dist.ravel()}")
    print(f"Calculated Focal Length: {focal_length_mm:.2f} mm (expected ~8 mm)")
    
    # Save numerical results
    np.savez('calibration_results/flir_calibration_final.npz',
             K=K, dist=dist, resolution=FLIR_RESOLUTION,
             rvecs=rvecs, tvecs=tvecs, per_image_errors=per_image_errors,
             pixel_pitch_mm=PIXEL_PITCH_MM)

    # Save undistortion example
    if image_paths:
        img = cv2.imread(image_paths[0])
        h, w = img.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1)
        undistorted = cv2.undistort(img, K, dist, None, new_K)
        comparison_img = np.hstack((img, undistorted))
        
        # Add text labels
        cv2.putText(comparison_img, "Original", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(comparison_img, "Undistorted", (w + 50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        cv2.imwrite('calibration_results/comparison_final.jpg', comparison_img)
        print("✅ Undistorted example saved to 'calibration_results/comparison_final.jpg'")
        
        # Print quality assessment
        print("\n=== CALIBRATION QUALITY ASSESSMENT ===")
        print(f"- {len(per_image_errors)}/{len(images)} images used in calibration")
        print(f"- Max image error: {max(per_image_errors):.3f} px")
        print(f"- Min image error: {min(per_image_errors):.3f} px")
        print(f"- Points with error >1px: {sum(e > 1 for e in all_errors)}/{len(all_errors)}")
        
        if ret < 0.5 and max(per_image_errors) < 1.0:
            print("✅ Calibration meets Euro NCAP requirements")
        else:
            print("⚠️ Calibration may not meet Euro NCAP requirements")
            print("   Review high-error images in reprojection_errors.png")


if __name__ == "__main__":
    calibrate_flir()