import cv2
import os
import time

# Khởi tạo camera
cam = cv2.VideoCapture(0)

# Sử dụng bộ phát hiện khuôn mặt từ OpenCV
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Nhập tên người dùng
while True:
    face_name = input(f'\n Nhập tên người dùng ==>  ').strip()
    file_path = 'dataset/labels.txt'

    if not os.path.exists('dataset'):
        os.mkdir('dataset')
        os.mkdir(f'dataset/{face_name}')
        break
    else:
        if not os.path.exists(f'dataset/{face_name}'):
            os.mkdir(f'dataset/{face_name}')
            break
        else:
            print('\n Đã tồn tại tên người dùng này, vui lòng sử dụng tên khác!')

print("\n [THÔNG BÁO] Đang khởi tạo tính năng chụp khuôn mặt. Nhìn vào camera và chờ đợi...")

# Khởi tạo biến lưu trữ số lượng ảnh khuôn mặt
count = 0

# Kiểm tra xem tệp có tồn tại hay chưa
file_exists = os.path.exists(file_path)

# Mở tệp nhãn với chế độ đẩy thêm (tạo tệp nếu tệp chưa tồn tại)
with open(file_path, 'a') as file:
    # Thêm ký tự dòng mới nếu tệp đã tồn tại
    if file_exists:
        file.write('\n')

    # Đẩy thêm nhãn vào cuối tệp
    text_to_append = face_name
    file.write(text_to_append)

# Các biến tính toán FPS
frame_count = 0
start_time = time.time()
fps = 0  # Khởi tạo biến FPS

# Bắt đầu vòng lặp chụp ảnh khuôn mặt
while True:
    # Đọc hình ảnh từ camera
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Lật hình ảnh video theo chiều dọc
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sử dụng bộ phát hiện khuôn mặt để tìm khuôn mặt trong ảnh
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Cắt và lưu ảnh chỉ chứa khuôn mặt
        gray = gray[y:y + h, x:x + w]

        try:
            cv2.imwrite("dataset/" + face_name + "/face_" + face_name + '_' + str(count) + ".jpg", gray)
        except Exception as e:
            continue

        # Tính toán FPS và hiển thị trên hình ảnh
        frame_count += 1
        if frame_count >= 1:  # Tính FPS sau mỗi 1 khung hình
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # Hiển thị FPS
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Hiển thị số lượng ảnh khuôn mặt đã chụp
        text_position = (img.shape[1] - 210, 30)
        cv2.putText(img, f"So anh: {count}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Image', img)
        count += 1

    # Nhấn 'ESC' để dừng máy ảnh
    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break
    elif count >= 500:  # Dừng chụp khi đủ 500 ảnh khuôn mặt
        break

print("\n [THÔNG BÁO] Đã hoàn thành quá trình thu thập dữ liệu!")

# Giải phóng tài nguyên camera và cửa sổ OpenCV
cam.release()
cv2.destroyAllWindows()
