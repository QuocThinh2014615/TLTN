import numpy as np
import argparse
import cv2
import os

# Cac tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, default='face_detector',
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=1,
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# Load model ssd nhan dien mat
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Kiểm tra xem thư mục đầu ra có tồn tại không, nếu không thì tạo mới
if not os.path.exists(args["output"]):
    os.makedirs(args["output"])

# Doc file video input
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# Lap qua cac frame cua video
while True:
    (grabbed, frame) = vs.read()
    # Neu khong doc duoc frame thi thoat
    if not grabbed:
        break

    read += 1
    if read % args["skip"] != 0:
        continue

    # Chuyen tu frame thanh blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Phat hien cac khuon mat trong frame
    net.setInput(blob)
    detections = net.forward()

    # Lap qua cac phat hien
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Neu muc do nhan dien > threshold
        if confidence > args["confidence"]:
            #Tach khuon mat va ghi ra file
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Đảm bảo rằng tọa độ nằm trong giới hạn của frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            # Cắt khuôn mặt từ frame
            face = frame[startY:endY, startX:endX]

            # Tạo đường dẫn file lưu trữ
            p = os.path.sep.join([args["output"], f"face_{saved:04d}.png"])
            cv2.imwrite(p, face)
            saved += 1
            print(f"[INFO] saved {p} to disk")

vs.release()
cv2.destroyAllWindows()

print(f"[INFO] Quá trình trích xuất hoàn thành. Đã lưu {saved} khuôn mặt vào {args['output']}.")
