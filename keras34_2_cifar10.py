from tensorflow.keras.datasets import cifar10     # 컬러 이미지 데이터
import numpy as np

filepath = 'C:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss: .4f}.hdf5'   # d: digit, f: float

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
print(date)
print(type(date))

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)     # reshape 할 필요 없음
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))
""" (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64)) """

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32, 32, 3), activation='relu'))     # (31, 31, 128)
model.add(Conv2D(filters=64, kernel_size=(2,2)))    # (30, 30, 64)
model.add(Conv2D(filters=32, kernel_size=(2,2)))    # (29. 29, 64)
model.add(Conv2D(filters=16, kernel_size=(2,2)))    # (28, 28, 32)
model.add(Flatten())    # 25,088 (= 28 * 28 * 32)
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath = filepath + 'k34_cifar10_' + date + filename)
                     # filepath = filepath + 'k34_cifar10_' + 'd_' + date + '_' + 'e_v_' + filename)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=25, validation_split=0.2, callbacks=[es, mcp])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


''' 32 X 32픽셀의 60,000개 컬러 이미지 포함
각 이미지는 10개의 클래스로 라벨링
60,000개의 이미지 중 50,000개는 train용, 10,000개는 test용
3channel: color 이미지
ReLu: 0 이상의 값들은 그대로 출력하게 하는 활성화 함수
Softmax: classification '''
