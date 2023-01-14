import numpy as np
from tensorflow.keras.datasets import mnist

# 경로 설정
# path = 'C:/study/_save/MCP/'
filepath = './_save/MCP/'
filename = '{epoch: 04d}-{val_loss: .4f}.hdf5'

# 파일 이름 설정 (날짜)
import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")   # 0114_1844

print(date)
print(type(date))                   # <class 'str'>
  
  
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)  -> input_shape = (28, 28, 1)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)     # 4차원으로 변경
x_test = x_test.reshape(10000, 28, 28, 1)

print(np.unique(y_train, return_counts=True))   # 배열 내 고유한 원소별 개수
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

#2. 모델
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(28, 28, 1), activation='relu'))   # (27, 27, 128)
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))     # (26, 26, 64)
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))     # (25, 25, 64)
model.add(Flatten())    # 40000 = 25 * 25 * 64
model.add(Dense(32, activation='relu'))     # input_shape = (60000, 40000)에서 '행 무시'이므로 (40000, ) 
                                            # 60000 = batch_size, 40000 = input_dim
model.add(Dense(10, activation='softmax'))  # 손글씨 이미지 분류 (숫자 0~9)
                                            # output 노드 10개이므로 다중 분류에 해당

#3. 컴파일, 훈련
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])


# earlystopping 설정
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

# modelcheckpoint 설정
mcp = ModelCheckpoint(monitor='val_loss', model='auto', verbose=1, save_best_only=True,
                      filepath = filepath + 'k34_mnist_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32, validation_split=0.25, callbacks=[es, mcp])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])

""" Epoch 00012: early stopping
313/313 [==============================] - 3s 8ms/step - loss: 0.0705 - acc: 0.9795
loss:  0.07049186527729034
acc:  0.9794999957084656 """

# kernel_size, filter, batch_size 등 변경해서 성능 향상 시킬 것

