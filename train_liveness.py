# train_liveness.py

import matplotlib
matplotlib.use("Agg")

from keras.utils import to_categorical  # Thêm dòng này
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from liveness_net import LivenessNet  # Cập nhật import từ liveness_net
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

# Tham số đầu vào
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output trained model")
ap.add_argument("-l", "--le", required=True, help="path to output label encoder")
ap.add_argument("-p", "--plot", required=True, help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# Khởi tạo tham số
INIT_LR = 1e-4  # Learning rate
BS = 8          # Batch size
EPOCHS = 50     # Số epochs để huấn luyện

# Khởi tạo dữ liệu và nhãn
data = []
labels = []

for category in ["real", "fake"]:
    imagePaths = os.listdir(os.path.join(args["dataset"], category))
    for imagePath in imagePaths:
        image = cv2.imread(os.path.join(args["dataset"], category, imagePath))
        image = cv2.resize(image, (32, 32))
        data.append(image)
        labels.append(category)

# Chuyển đổi dữ liệu và nhãn thành numpy array
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# One-hot encoding nhãn
le = LabelBinarizer()
labels = le.fit_transform(labels)

# Kiểm tra hình dạng của nhãn
print("Shape of labels after encoding:", labels.shape)

# Chia dữ liệu thành train và test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Chuyển đổi nhãn sang định dạng one-hot
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# Kiểm tra hình dạng của nhãn sau khi chuyển đổi
print("Shape of trainY:", trainY.shape)
print("Shape of testY:", testY.shape)

# Tạo đối tượng ImageDataGenerator để augment dữ liệu
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Khởi tạo mô hình
print("[INFO] compiling model...")
model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
opt = Adam(learning_rate=INIT_LR)  # Cập nhật ở đây
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# Huấn luyện mô hình
print("[INFO] training network...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS)

# Đánh giá mô hình
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# Lưu mô hình và label encoder
model_filepath = args["model"]
if not model_filepath.endswith(('.h5', '.keras')):
    model_filepath += '.h5'  # Hoặc '.keras'

print("[INFO] saving model and label encoder...")
model.save(model_filepath)
with open(args["le"], "wb") as f:
    f.write(pickle.dumps(le))

# Vẽ đồ thị loss và accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
