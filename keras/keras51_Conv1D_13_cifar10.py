import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))
""" (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64)) """

x_train = x_train.reshape(50000, 32*3, 32)
x_test = x_test.reshape(10000, 32*3, 32)

x_train = x_train/255
x_test = x_test/255

#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(32*3, 32)))
model.add(Conv1D(64, 2, input_shape=(32*3, 32)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))      # output 노드가 10개이므로 다중 분류

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=10,
                              restore_best_weights=True,
                              verbose=1)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=10, batch_size=25,
                 validation_split=0.2, callbacks=[EarlyStopping], verbose=1)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])


""" loss :  1.7157658338546753
acc :  0.38499999046325684 """