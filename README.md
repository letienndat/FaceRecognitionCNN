## FACE RECOGNITION CNN

## Yêu cầu
### Nâng cấp pip
```
pip install --upgrade pip
```
### Cài đặt thư viện
#### OpenCV
```
pip install opencv-python
pip install opencv-contrib-python
```
#### Tensorflow and Keras (Yêu cầu Python 3.9 -> 3.11)
```
pip install tensorflow
```
#### Sklearn
```
pip install scikit-learn
```
#### Matplotlib
```
pip install matplotlib
```
#### Glob
```
pip install glob2
```

## CHẠY CHƯƠNG TRÌNH
### Bước 1
- Chạy file 01_face_dataset.py -- Thao tác này sẽ chụp 500 ảnh khuôn mặt của bạn và lưu nó vào thư mục dataset
### Bước 2
- Chạy file 02_face_training.py -- File này sẽ huấn luyện mô hình CNN và lưu trọng số vào file 'face_detection.keras'
### Bước 3
- Chạy file 03_face_recognition.py -- Tự động mở webcam lên và tiến hành nhận diện khuôn mặt