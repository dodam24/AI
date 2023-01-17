# [실습]
# R2 0.55 ~ 0.6 이상

import numpy as np
import sklearn as sk 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x)
print(x.shape)  # (20640, 8)   # 특성이 8개 (input_dim = 8)
print(y)
print(y.shape)  # (20640, 1)   # 마지막 layer의 값(output) = 1

print(dataset.feature_names)

# print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123)

#2. 모델 구성
model=Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
start = time.time()
model.fit(x_train, y_train, epochs=3000, batch_size=32)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

print('time: ', end-start)

# 결과 : R2 :  0.5961615312635271