# [실습]
# 1. train 0.7 이상
# 2. R2 : 0.8 이상 / RMSE 사용

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import sklearn as sk
print(sk.__version__)   # 1.1.3

from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()     # boston의 부동산 관련 데이터 (교육용 데이터셋 불러오기)
x = dataset.data            # house data
y = dataset.target          # house price

print(x)
print(x.shape)  # (506, 13)
print(y)
print(y.shape)  # (506,)

print(dataset.feature_names)    # data column name
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(dataset.DESCR)

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8, shuffle=True, random_state=123)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(60))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
# model = LinearRegression()
model.compile(loss='mse', optimizer='adam',
            metrics=['mae'])
model.fit(x_train, y_train, epochs=5000, batch_size=32)   # batch_size의 default=32

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결과 : 