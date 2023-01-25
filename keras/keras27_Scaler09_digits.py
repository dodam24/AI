import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

# 데이터 전처리
scaler = MinMaxScaler()                     # minmaxscaler 정의
# scaler = StandardScaler()
scaler.fit(x_train)                         # x값의 범위만큼 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)           # x_train fit한 가중치 값 범위에 맞춰서 x_test 데이터 변환

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(64,)))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(100, activation='linear'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# EarlyStopping 적용, 시간 측정
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)
import time
start = time.time()
model.fit(x_train, y_train, epochs=700, batch_size=32, validation_split=0.2, verbose=1, callbacks=[earlyStopping])
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



""" Epoch 00148: early stopping
loss :  1.1648075580596924
accuracy :  0.7666666507720947
y_pred :  [6 3 6 7 7 6 7 1 0 2 6 6 1 6 3 0 3 5 7 0 5 1 3 2 4 7 9 7 5 6 4 8 1 0 9 5 7
 4 6 1 7 0 0 3 1 1 3 3 0 7 6 5 1 0 7 9 7 5 9 6 3 6 2 5 0 6 1 4 7 6 6 5 7 6
 0 0 5 3 9 2 5 4 4 3 3 2 4 9 2 2 6 4 1 5 4 0 1 1 5 0 6]
y_test :  [6 3 6 7 7 1 7 1 0 2 6 6 1 6 3 0 5 5 7 0 5 1 5 2 4 9 5 7 5 1 4 8 1 0 9 5 9
 4 8 1 7 0 0 3 1 7 3 3 0 8 8 5 4 0 7 8 7 5 5 6 3 8 2 9 0 8 2 4 7 8 8 5 7 8
 0 0 5 2 9 2 5 4 4 3 3 2 4 9 2 2 1 4 1 5 4 0 7 1 5 0 8]
accuracy :  0.7666666666666667
time :  5.672308444976807 """