import numpy as np
import cv2
import time

video_name = "street_1.mp4"
root_dir = "D:\\Pedestrian Detection\\dnn_detector"
proto = root_dir + "\\MobileNetSSD_deploy.prototxt.txt"
mobile_ssd = root_dir + "\\MobileNetSSD_deploy.caffemodel"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = (0, 255, 0)
confidence_threshold = 0.2

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(proto, mobile_ssd)
cap = cv2.VideoCapture(video_name)

while cap.isOpened():
    tic = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, size=(300, 300), mean=127.5)
    net.setInput(blob)
    detections = net.forward()
    detections = detections[0, 0, :]

    detections = np.asarray([det[2:] for det in detections if int(det[1]) == CLASSES.index('person') and det[2] > confidence_threshold])

    if detections.size > 0:
        Height_Analysis_range = detections[:, [2, 4]] * h
        Height_Analysis_range = Height_Analysis_range.astype("int")
        Height_range = Height_Analysis_range[:, 1] - Height_Analysis_range[:, 0]
        Height_median = np.median(Height_range)
        height_scale = 0.2

        for det in detections:
            confidence, box = det[0], det[1:] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            label = f"person:{confidence * 100:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS, 2)
            y = y1 - 15 if y1 - 15 > 15 else y1 + 15
            cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS, 2)

            if y2 - y1 < int(Height_Analysis_range[0][0] * (1 + 2 * 0.2)):
                if y2 - y1 > Height_median * (1 + height_scale):
                    Height_label = "High"
                elif Height_median * (1 - height_scale) < y2 - y1 < Height_median * (1 + height_scale):
                    Height_label = "Medium"
                else:
                    Height_label = "Short"
                cv2.putText(frame, Height_label, (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (211, 211, 211), 2)

    toc = time.time()
    durr = toc - tic
    fps = 1.0 / durr
    cv2.putText(frame, f"fps:{fps:.3f}", (20, 20), 3, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
