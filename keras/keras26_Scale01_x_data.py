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

# scaler = MinMaxScaler()       # minmaxscaler 정의
scaler = StandardScaler()
scaler.fit(x)                   # x값의 범위만큼 가중치 생성. x_train 데이터 넣기
x = scaler.transform(x)         # fit으로 학습시킨 후, transform 사용하여 변환
print(x)                        # 최소 ~ 최대가 0 ~ 1로 모든 숫자들이 정규화 됨
print(type(x))

# print("최솟값 : ", np.min(x))
# print("최댓값 : ", np.max(x))

"""
# ############### MinMaxScaler ###############      # 정규화: 모든 데이터를 0~1 사이로 변환
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print(x)
print(x.shape)
print(type(x))
print(np.min(x))
print(np.max(x))
# ############################################ """

############### Standard Scaler ###############     # 표준화: 모든 데이터를 평균 0, 분산 1인 정규분포로 변경
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
###############################################

# print(x)
print(x.shape)   # (506, 13)
# print(y)
print(y.shape)   # (506,)

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.3, shuffle=True, random_state=123)


#2. 모델 구성
model = Sequential()
# model.add(Dense(10, input_dim=13, ))          # input_dim은 행, 열 형태일 때만 가능
model.add(Dense(10, input_shape=(13, )))        # (13, ) 형태는 input_shpae 적용
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


""" 
loss :  [64.87870788574219, 5.8619842529296875]
RMSE :  8.054731976120436
R2 :  0.26462716027999067 """



"""
데이터 스케일링: 
피처(feature)들마다 데이터값의 범위가 다 제각각이기 때문에
범위 차이가 클 경우, 데이터를 가지고 모델을 학습할 때 
0으로 수렴하거나 무한으로 발산할 가능성이 있음
따라서, 데이터 스케일링을 통해 모든 피처들의 데이터 분포나 범위를 동일하게 조정해야 함
 = '정규화' 또는 '표준화'를 이용하여 해결 가능!
-> 객체를 생성하여 학습(fit)시킨 후, transform으로 변환


1. MinMaxScaler(정규화)
모든 데이터를 0~1 사이의 값으로 바꾸는 것
if 최댓값 10, 최솟값이 2일 때, 4는 0.25로 정규화됨. (4-2) / (10-2) = 0.25
(X - (X의 최솟값)) / (X의 최댓값 - X의 최솟값) 


2. StandardScaler(표준화)
모든 데이터를 평균이 0, 분산이 1인 정규 분포로 변경
if 평균이 50, 임의의 숫자가 49일 경우, 평균으로부터 얼마나 떨어져 있는지 구하는 것. (49-50) / 1 = -1
(Xi - (X의 평균)) / (X의 표준편차)
"""
