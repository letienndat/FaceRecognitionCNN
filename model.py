from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras import backend as K


# Định nghĩa mô hình
def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":  # Kiểm tra định dạng dữ liệu hình ảnh, channels_first hoặc channels_last
        inputShape = (depth, height, width)
        chanDim = 1

    # Thêm lớp Conv2D với 32 bộ lọc kích thước (3, 3), padding "same" để giữ nguyên kích thước đầu vào
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))  # Lớp MaxPooling2D với kích thước cửa sổ (3, 3)
    model.add(Dropout(0.25))  # Lớp Dropout với tỷ lệ loại bỏ 25%

    # Thêm lớp Conv2D với 64 bộ lọc, tiếp theo là Activation, BatchNormalization, và MaxPooling2D
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Lặp lại quá trình với Conv2D, Activation, BatchNormalization, MaxPooling2D, và Dropout
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())  # Biến đổi dữ liệu hình ảnh thành dạng mảng 1D
    model.add(Dense(1024))  # Lớp Dense với 1024 đơn vị
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))  # Lớp cuối cùng với số lượng đơn vị bằng số lượng lớp classes (số nhãn)
    model.add(Activation("sigmoid"))  # Hàm kích hoạt Sigmoid cho phân loại nhị phân

    return model
