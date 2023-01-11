import numpy as np
import pandas as pd
import sklearn as sk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

# scaler = MinMaxScaler() # minmaxscaler 정의
scaler = StandardScaler()
scaler.fit(x) # x값의 범위만큼의 가중치 생성 / x_train 데이터 넣기
x = scaler.transform(x) # 시작 (transform해야 바뀐다.)
print(x)
print(type(x))

# print("최소값 : ", np.min(x))
# print("최대값 : ", np.max(x))

# ########## MinMaxScaler ##########
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)

# print(x)
# print(x.shape)
# print(type(x))
# print(np.min(x))
# print(np.max(x))
# ##################################

########## Standard Scaler ##########
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
#####################################

# print(x)
print(x.shape)   # (506, 13)
# print(y)
print(y.shape)   # (506,)

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']
print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.3, shuffle=True, random_state=123)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.3, shuffle=True, random_state=123)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13, activation = 'linear'))
model.add(Dense(7, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
            metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결과
# loss :  [64.87870788574219, 5.8619842529296875]
# RMSE :  8.054731976120436
# R2 :  0.26462716027999067
