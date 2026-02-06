# Import required modules
import cv2
import numpy as np

# ================= CONFIG =================
CHECKERBOARD = (7, 9)          # inner corners (cols, rows)
SQUARE_SIZE = 25.0             # mm
ESP32_STREAM_URL = "http://10.54.102.56:81/stream"  # <-- CHANGE IP
# =========================================

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# Storage for calibration points
threedpoints = []
twodpoints = []

# Prepare real-world object points
objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                               0:CHECKERBOARD[1]].T.reshape(-1, 2)
objectp3d *= SQUARE_SIZE   # 🔴 25 mm squares applied here

# Open ESP32 stream
cap = cv2.VideoCapture(ESP32_STREAM_URL)

if not cap.isOpened():
    print("❌ Cannot open ESP32 stream")
    exit()

print("✅ SPACE = capture frame | ESC = calibrate & exit")

last_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    display = frame.copy()

    if found:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners2, found)

    cv2.imshow("ESP32 Camera Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    # ESC → finish
    if key == 27:
        break

    # SPACE → capture
    if key == 32 and found:
        threedpoints.append(objectp3d)
        twodpoints.append(corners2)
        last_gray = gray.copy()
        print(f"📸 Captured frame {len(threedpoints)}")

cap.release()
cv2.destroyAllWindows()

# ================= CALIBRATION =================
if len(threedpoints) < 10:
    print("❌ Not enough frames captured (need ≥10)")
    exit()

ret, camera_matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints,
    twodpoints,
    last_gray.shape[::-1],
    None,
    None
)

print("\n✅ Calibration successful\n")

print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", distortion)
print("\nRotation Vectors:\n", r_vecs)
print("\nTranslation Vectors (mm):\n", t_vecs)

