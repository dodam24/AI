import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (1797, 64) (1797,)
print(np.unique(y, return_counts=True))     # array에서 각 고유(unique) 원소의 개수 출력
""" 
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),     # 다중 분류 데이터
array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64)) """

# 시각화
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[5])
plt.show()

# One-hot 인코딩 통해서 10진 정수 형식을 2진 바이너리 형식으로 변경 (0, 1)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)      # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=333)

print(x_train.shape, x_test.shape)      # (1437, 64) (360, 64)

x_train = x_train.reshape(1437, 64, 1)       
x_test = x_test.reshape(360, 64, 1)
print(x_train.shape, x_test.shape)


#2. 모델 구성 (순차형)
model = Sequential()
model.add(LSTM(64, (2,2), input_shape=(4, 4, 4)))
model.add(Dense(10, activation='linear'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# EarlyStopping 적용, 시간 측정
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)
import time
start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1, callbacks=[earlyStopping])
end = time.time()

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)    # np.argmax: 최대값의 위치(index) 반환
y_test = np.argmax(y_test, axis=1)
print('y_pred : ', y_predict)               # 예측값
print('y_test : ', y_test)                  # 원래 값

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)     # 정확도: 실제 데이터 중 맞게 예측한 데이터의 비율
print('accuracy : ', acc)
print('time : ', end - start)

