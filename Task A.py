import cv2
import numpy as np
import matplotlib.pyplot as plt

# === DISABLE OpenCL to avoid memory errors ===
cv2.ocl.setUseOpenCL(False)

# === PARAMETERS ===
BRIGHTNESS_THRESHOLD = 100

# === USER INPUT ===
video_path = input("Enter video path (e.g., 'street.mp4'): ").strip()

# === OPEN VIDEO ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# === READ FIRST 50 FRAMES TO CHECK BRIGHTNESS ===
brightness_values = []
for _ in range(50):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    brightness_values.append(brightness)

cap.release()

# === CHECK IF FRAMES WERE READ ===
if not brightness_values:
    print("Error: No frames read from the video.")
    exit()

# === CALCULATE AVERAGE BRIGHTNESS ===
average_brightness = np.mean(brightness_values)
print("Average Brightness:", average_brightness)

# === CLASSIFY DAY OR NIGHT ===
if average_brightness >= BRIGHTNESS_THRESHOLD:
    print("This is likely a daytime video.")
else:
    print("This is likely a nighttime video.")

# === PLOT HISTOGRAM OF BRIGHTNESS ===
plt.figure(figsize=(8, 5))
plt.hist(brightness_values, bins=20, color='blue', edgecolor='black')
plt.title('Brightness Distribution (First 50 Frames)')
plt.xlabel('Average Brightness per Frame')
plt.ylabel('Number of Frames')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Watermark ===
def brighten(path):
    wm = cv2.imread(path).astype(float)
    return np.clip(wm * 1.8, 0, 255).astype(np.uint8)

wm1 = brighten("watermark1.png")
wm2 = brighten("watermark2.png")
h_wm, w_wm, _ = wm1.shape

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

# === Apply watermark and process ===
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Position: center
    center_y = height // 2
    center_x = width // 2
    top_y = center_y - h_wm // 2
    left_x = center_x - w_wm // 2
    bottom_y = top_y + h_wm
    right_x = left_x + w_wm

    # Get Roi
    roi = frame[top_y:bottom_y, left_x:right_x]

    current_wm = wm1 if (frame_idx // (fps * WM_SWITCH_INTERVAL)) % 2 == 0 else wm2
    blended = cv2.addWeighted(roi, 1.0, current_wm, WM_ALPHA, 0)
    frame[top_y:bottom_y, left_x:right_x] = blended

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
# === Watermark Fin ===