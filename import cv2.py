import cv2
import numpy as np
import os

# === File Paths ===
folder_path = r"D:\Drone cia"  # Folder where yolov4-tiny files are
config_path = os.path.join(folder_path, "yolov4-tiny.cfg")
weights_path = os.path.join(folder_path, "yolov4-tiny.weights")
names_path = os.path.join(folder_path, "coco.names")
video_path = r"video.mp4"  # Input video
output_path = r"output_red_jacket.mp4"  # Output video

# === Load class labels ===
with open(names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# === Load YOLOv4-Tiny ===
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# === Open Video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# === Output Video Writer ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# === Red Color Range in HSV ===
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150])
upper_red2 = np.array([180, 255, 255])

print("🔍 Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete.")
        break

    height, width = frame.shape[:2]

    # === YOLO forward pass ===
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Detect only PERSON (class_id == 0 in COCO dataset)
            if class_id == 0 and confidence > 0.6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.5)

    for i in indexes.flatten():
        x, y, w, h = boxes[i]

        # Crop person ROI
        person_roi = frame[y:y + h, x:x + w]
        if person_roi.size == 0:
            continue

        # === Focus only on upper 50% of body (torso region) ===
        upper_body = person_roi[0:int(h * 0.5), :]
        if upper_body.size == 0:
            continue

        hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)

        # Red mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        # Morphological filter to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        red_pixels = cv2.countNonZero(mask)
        total_pixels = upper_body.shape[0] * upper_body.shape[1]

        # If >5% of torso is red => assume jacket
        if total_pixels > 0 and (red_pixels / total_pixels > 0.05):
            label = "ALERT: Red Jacket"
            color = (0, 0, 255)  # Red
        else:
            label = "Person"
            color = (0, 255, 0)  # Green

        # Draw bounding box + label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    out.write(frame)
    cv2.imshow("Red Jacket Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()