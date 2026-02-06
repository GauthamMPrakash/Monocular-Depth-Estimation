import cv2
import matplotlib
import numpy as np
import torch
import os
from depth_anything_v2.dpt import DepthAnythingV2

# -----------------------------
# CONFIG
# -----------------------------
STREAM_URL = "http://172.16.8.114:8080/video"  # Your ESP32 HTTP MJPEG stream
INPUT_SIZE = 518
OUTDIR = "./esp32_depth"
ENCODER = 'vits'  # must match your checkpoint
CHECKPOINT = "checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
MAX_DEPTH = 20
SAVE_NUMPY = False
PRED_ONLY = False
GRAYSCALE = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(OUTDIR, exist_ok=True)
cmap = matplotlib.colormaps.get_cmap('Spectral')

# -----------------------------
# CAMERA CALIBRATION
# -----------------------------
# Replace these with your ESP32 calibration values
K = np.array([[287.30455698, 0, 169.77191062],
              [0, 299.48090876, 100.60857848],
              [0,  0,  1]])
dist = np.array([0, 0, 0, 0, 0])  # distortion coefficients

def undistort(frame):
    h, w = frame.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w,h), cv2.CV_32FC1)
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

# -----------------------------
# MODEL CONFIG
# -----------------------------
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_anything = DepthAnythingV2(**{**model_configs[ENCODER], 'max_depth': MAX_DEPTH})
depth_anything.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()

# -----------------------------
# OPEN HTTP STREAM
# -----------------------------
cap = cv2.VideoCapture(STREAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError("Cannot open ESP32 HTTP stream")

print("✅ ESP32 HTTP stream opened")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received, retrying...")
        continue

    # -----------------------------
    # UNDISTORT FRAME
    # -----------------------------
    frame = undistort(frame)

    # -----------------------------
    # RUN DEPTH ESTIMATION
    # -----------------------------
    depth = depth_anything.infer_image(frame, INPUT_SIZE)

    # -----------------------------
    # COMPUTE NEAREST DISTANCE USING INTRINSICS
    # -----------------------------
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth  # depth in meters
    x = (i - K[0, 2]) * z / K[0, 0]
    y = (j - K[1, 2]) * z / K[1, 1]
    points_3d = np.stack([x, y, z], axis=-1)
    nearest_distance = np.min(points_3d[..., 2])
    print(f"Frame {frame_count}: Nearest object = {nearest_distance:.2f} m")

    if SAVE_NUMPY:
        output_path = os.path.join(OUTDIR, f'frame_{frame_count:05d}_raw_depth_meter.npy')
        np.save(output_path, depth)

    # normalize for visualization
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_vis = depth_vis.astype(np.uint8)

    if GRAYSCALE:
        depth_vis = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
    else:
        depth_vis = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    # combine with RGB if PRED_ONLY is False
    if not PRED_ONLY:
        split_region = np.ones((frame.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([frame, split_region, depth_vis])
    else:
        combined_result = depth_vis

    # overlay nearest distance on screen
    cv2.putText(combined_result, f"Nearest: {nearest_distance:.2f} m", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # display
    cv2.imshow("ESP32 Metric Depth", combined_result)
    frame_count += 1

    # exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

