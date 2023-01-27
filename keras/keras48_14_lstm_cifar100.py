from tensorflow.keras.datasets import cifar100   # 컬러 이미지를 분류 (100개의 클래스)
import numpy as np

filepath = 'C:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss: .4f}.hdf5'

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   # 0114_2235
print(date)
print(type(date))                   # <class 'str'>


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))
""" (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
       51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
       68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
       85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), 
       array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64)) """
       
# 100개의 클래스로 분류되며, 각각의 클래스는 600개의 이미지(500개의 훈련 데이터, 100개의 테스트 데이터)이며,
# 총 60,000개의 이미지로 구성된 데이터셋인 CIFAR-100 컬러 이미지를 분류하는 CNN 구현


#2. 모델    # padding, maxpooling, strides 추가
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

model = Sequential()
model.add(LSTM(units=128, input_shape=(32, 32, 3)))
model.add(Dense(8, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

#. 컴파일, 훈련
''' from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=3)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=3, save_best_only=True,
                      filepath = filepath + 'k34_cifar100_' + 'd_' + date + '_' + 'e_v_' + filename) '''

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, verbose=3, batch_size=32, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
