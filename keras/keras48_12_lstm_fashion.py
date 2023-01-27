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
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

model = Sequential()
model.add(LSTM(units=128, input_shape=(, )))
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