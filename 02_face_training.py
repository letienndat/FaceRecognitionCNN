from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

from model import build  # Import mô hình được xây dựng từ file 'model.py'

# Các thông số ban đầu
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96, 96, 3)

data = []
labels = []

# Nạp tệp ảnh từ tập dữ liệu
image_files = [f for f in glob.glob(r'dataset' + "/**/*", recursive=True) if
               not os.path.isdir(f) and f != 'dataset\labels.txt']
random.shuffle(image_files)

# Mở tệp 'labels.txt' để đọc nhãn
with open('dataset/labels.txt', 'r') as file:
    # Đọc toàn bộ nội dung của tệp và tách thành danh sách các nhãn
    _labels = file.read().split('\n')

# Chuyển đổi hình ảnh thành mảng và gán nhãn cho các loại
for img in image_files:
    image = cv2.imread(img)

    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]
    label = _labels.index(label)

    labels.append([label])

# Tiền xử lý dữ liệu
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)

trainY = to_categorical(trainY, num_classes=len(_labels))  # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(testY, num_classes=len(_labels))

# Tạo dữ liệu mới bằng cách biến đổi tập dữ liệu hiện có
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# Xây dựng mô hình
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
              classes=len(_labels))

# Biên dịch mô hình
opt = Adam(learning_rate=lr, weight_decay=lr / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Huấn luyện mô hình
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=epochs,
    verbose=1
)

# Lưu mô hình xuống ổ đĩa
model.save('D:/face_detection.keras')

# Vẽ biểu đồ về sự biến thiên của loss và accuracy trong quá trình huấn luyện
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Sự biến thiên của Loss và Accuracy trong quá trình Huấn luyện")
plt.xlabel("Số epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# Lưu biểu đồ
plt.savefig('plot.png')