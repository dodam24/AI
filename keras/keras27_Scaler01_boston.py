import numpy as np
# import pandas as pd
# # import sklearn as sk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.3, shuffle=True, random_state=123)


scaler = MinMaxScaler()                     # minmaxscaler 정의
# scaler = StandardScaler()
scaler.fit(x_train)                         # x값의 범위만큼 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)           # 시작 (transform해야 바뀐다.)
                                            # x_train fit한 가중치 값 범위에 맞춰서 x_test 데이터 변환

""" 
sklearn으로 데이터 스케일링을 할 경우,
훈련(train)용 데이터셋과 테스트(test)용 데이터셋에 각각 다른 과정이 필요함
 - 훈련(train)용 데이터셋은 fit 메서드를 적용시킨 후, transform 해주어야 하고,
 - 테스트(test)용 데이터셋에는 transform을 적용시켜주면 됨 """


# print(x)
print(x.shape)   # (506, 13)
# print(y)
print(y.shape)   # (506,)

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(dataset.DESCR)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(7))
model.add(Dense(2))
model.add(Dense(5))
model.add(Dense(70))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
            metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=32)

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