import pickle

import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils, print_summary
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Khai báo đường dẫn lưu file history chứa thông tin về quá trình train
file_history = "history/history.npy"
filepath = "QuickDraw.h5"  # đường dẫn file weights
epochs = 15  # xác định số epochs
train_x, test_x, train_y, test_y = [], [], [], []

def keras_model(image_x, image_y):
    num_of_classes = 4

    # Định nghĩa mô hình Keras: là mô hình tuần tự các lớp ANN `Sequential Model`
    # và nạp tất cả các Layers của ANN một lần vào mô hình đó
    model = Sequential()

    # Tầng Input
    # convolution layer dùng để lấy feature từ image
    # model.add(Conv2D(32, (5, 5), input_shape=(image_x,image_y,1), activation='relu'))
    # Gồm 32 filters, kích thước mỗi filter là (3,3) => tổng số weights = 32*3*3 = 288
    model.add(Conv2D(32, (3, 3), input_shape=(image_x, image_y, 1), activation='relu'))
    # dùng để lấy feature nổi bật(dùng max) và giúp giảm parameter khi training
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # convolution layer dùng để lấy feature từ image
    # model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # dùng để lấy feature nổi bật(dùng max) và giúp giảm parameter khi training
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # Lát phằng layer để fully connection, vd: shape 20x20 qua layer này sẽ là 400x1
    model.add(Flatten())
    # Fully connected layers
    model.add(Dense(512, activation='relu'))
    # layer này dùng như regularization cho các layer để hạn chế overfiting
    # rate = 0.5 => 50% neuron bị loại đi
    model.add(Dropout(rate=0.25))
    # Fully connected layers
    model.add(Dense(128, activation='relu'))
    # layer này dùng như regularization cho các layer để hạn chế overfiting
    # rate = 0.5 => 50% neuron bị loại đi
    model.add(Dropout(rate=0.25))

    # Tầng output
    # Fully connected layers, softmax dùng trong multi classifier
    model.add(Dense(num_of_classes, activation='softmax'))

    # Cấu hình mô hình để train
    # categorical_crossentropy: dùng trong classifier nhiều class
    # metrics: thước đo để ta đánh giá accuracy của model
    # optimizers: dùng để chọn thuật toán training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Khi model chúng ta lớn có khi training thì gặp sự cố ta muốn lưu lại model để chạy lại thì callback giúp ta làm điều này.
    # ModelCheckpoint lưu lại model sau mỗi epoch
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))
    return features, labels

def prepress_labels(labels):
    ''' Sử dụng one-hot encoding để xử lý dữ liệu, Converts một class vector (gồm các số nguyên) sang ma trận class nhị phân. '''
    labels = np_utils.to_categorical(labels)
    return labels

def VisualizeTrainingHistory():
    ''' Vẽ đồ thị accuracy và loss '''

    # load history
    history = np.load(file_history, allow_pickle='TRUE').item()
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def trainModel():
    ''' Hàm phụ trách train model keras được định nghĩa qua hàm keras_maodel(x, y) '''
    # Load features, labels từ files đã lưu
    features, labels = loadFromPickle()
    # Xáo trộn tập dữ liệu
    features, labels = shuffle(features, labels)
    # convert class vector
    labels = prepress_labels(labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state = 0, test_size = 0.2)
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
    # Khởi tạo model keras
    model, callbacks_list = keras_model(28, 28)
    # In bảng tóm tắt kiến trúc model.
    print_summary(model)
    # Hiệu chỉnh mô hình
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=64, callbacks=[TensorBoard(log_dir="QuickDraw")])
    # Lưu lại mô hình đã train
    model.save(filepath)
    # save history
    np.save(file_history, model.history)

    return model


if __name__ == "__main__":
    model = trainModel()
    VisualizeTrainingHistory()

    