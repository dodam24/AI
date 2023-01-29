import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout

#1. 데이터
# 텐서플로 mnist 데이터셋 불러와서 변수에 저장하기
# datasets = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)  ->  input_shape = (28, 28, 1)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)


# output class 개수 확인
print(np.unique(y_train, return_counts=True))
""" 
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64)) 
"""


#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(28, 28)))
model.add(Conv1D(64, 2, input_shape=(28, 28)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))      # 손글씨 이미지 분류 (0~9).     output 노드가 10개이므로 다중 분류에 해당

model.summary()


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=10,
                              restore_best_weights=True,
                              verbose=1)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=25,
                 validation_split=0.2, callbacks=[EarlyStopping], verbose=1)


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


""" loss :  2.1714165210723877
acc :  0.15919999778270721 """