from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2

# Sử dụng CascadeClassifier để tải bộ phân loại khuôn mặt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Tải mô hình nhận dạng khuôn mặt đã được huấn luyện
model = load_model('D:/face_detection.keras')

# Mở webcam
webcam = cv2.VideoCapture(0)

with open('dataset/labels.txt', 'r') as file:
    # Đọc toàn bộ nội dung của tệp 'labels.txt' chứa các nhãn người dùng
    classes = file.read().split('\n')

# Lặp qua từng khung hình từ webcam
while webcam.isOpened():

    # Đọc khung hình từ webcam
    status, frame = webcam.read()

    # Áp dụng phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    # Lặp qua các khuôn mặt đã được phát hiện
    for idx, f in enumerate(faces):

        # Lấy tọa độ góc của hình chữ nhật khuôn mặt
        (x, y, w, h) = f[0], f[1], f[2], f[3]

        # Vẽ hình chữ nhật trên khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cắt vùng khuôn mặt đã phát hiện
        face_crop = np.copy(frame[y:y + h, x:x + w])

        # Kiểm tra kích thước của vùng khuôn mặt
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Thay đổi kích thước vùng khuôn mặt
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Áp dụng mô hình nhận dạng khuôn mặt
        conf = model.predict(face_crop)[0]  # model.predict trả về ma trận 2D, ví dụ: [[9.9993384e-01 7.4850512e-05]]

        print(conf)

        # Lấy nhãn có độ chính xác cao nhất
        index = np.argmax(conf)
        label = classes[index]

        # Xác định nhãn là "Unknown" nếu độ chính xác thấp hơn 0.7, ngược lại hiển thị nhãn và độ chính xác
        if conf[index] < 0.7:
            label = 'Unknown'
        else:
            label = "{}: {:.2f}%".format(label, conf[index] * 100)

        # Tọa độ Y để viết nhãn và độ chính xác lên khung hình khuôn mặt
        Y = y - 10 if y - 10 > 10 else y + 10

        # Hiển thị nhãn và độ chính xác trên khung hình
        cv2.putText(frame, label, (x, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow("Face Detection", frame)

    # Nhấn phím "ESC" để dừng
    if cv2.waitKey(20) & 0xff == 27:
        break

# Giải phóng tài nguyên
webcam.release()
cv2.destroyAllWindows()