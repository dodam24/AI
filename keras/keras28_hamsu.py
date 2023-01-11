import numpy as np
# import pandas as pd
# # import sklearn as sk
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.3, shuffle=True, random_state=123)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.3, shuffle=True, random_state=123)

scaler = MinMaxScaler() # minmaxscaler 정의
#  scaler = StandardScaler()
scaler.fit(x_train) # x값의 범위만큼의 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)# 시작 (transform해야 바뀐다.)



# print(x)
print(x.shape)   # (506, 13)
# print(y)
print(y.shape)   # (506,)

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']
print(dataset.DESCR)



#2. 모델 구성(순차형) # model 형태를 맨 처음에 정의
# model = Sequential()
# model.add(Dense(50, input_dim=13))
# model.add(Dense(40, activation='linear'))
# model.add(Dense(30, activation='sigmoid'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(1, activation='linear'))
# model.summary()
# total param : 

#2. 모델 구성(함수형) # 순차형과 반대로 레이어 구성
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='linear')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(10, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1) # 순차형과 달리 model 형태를 마지막에 정의
model.summary()

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

# r2 = r2_score(y_test, y_predict)
# print("R2 : ", r2)

# # 결과
# # loss :  [64.87870788574219, 5.8619842529296875]
# # RMSE :  8.054731976120436
# # R2 :  0.26462716027999067
