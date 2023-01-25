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


# 데이터 전처리
scaler = MinMaxScaler()                     # minmaxscaler 정의
# scaler = StandardScaler()
scaler.fit(x_train)                         # x값의 범위만큼 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)           # x_train fit한 가중치 값 범위에 맞춰서 x_test 데이터 변환


# print(x)
print(x.shape)   # (506, 13)
# print(y)
print(y.shape)   # (506,)

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(dataset.DESCR)


#2. 모델 구성 (순차형)       # 모델 형태를 맨 처음에 명시
# model = Sequential()
# model.add(Dense(50, input_dim=13))
# model.add(Dense(40, activation='linear'))
# model.add(Dense(30, activation='sigmoid'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(1, activation='linear'))
# model.summary()

#2. 모델 구성 (함수형)                                  # 순차형과 반대로 레이어 구성
input1 = Input(shape=(13,))                             # 입력 데이터의 크기(shape)를 Input() 함수의 인자로 입력층 정의
dense1 = Dense(50, activation='linear')(input1)         # 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)           # 순차형과 달리 model 형태를 마지막에 정의.     Model() 함수에 입력과 출력 정의
model.summary()
""" 
Model: "model"      # Param 수가 더 많은 이유: bias 때문
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 13)]              0

 dense (Dense)               (None, 50)                700              13 * 50 + 50 = 700                  

 dense_1 (Dense)             (None, 40)                2040             50 * 40 + 40 = 2040

 dense_2 (Dense)             (None, 30)                1230             40 * 30 + 30 = 1230

 dense_3 (Dense)             (None, 20)                620              30 * 20 + 20 = 620

 dense_4 (Dense)             (None, 1)                 21               20 * 1 + 1 = 21

=================================================================
Total params: 4,611
Trainable params: 4,611
Non-trainable params: 0 """

# Output shape = (None, 50) None개의 Input과 50개의 output을 의미
# Param: 입력 노드와 출력 노드 사이에 연결된 간선의 수
# input node 수 * output node 수 + output node 수(=bias node)
# Bias Node: 입력 패턴의 모든 값이 0이면 가중치가 변하지 않아 모델이 학습할 수 없으므로 가상의 Input을 생성
#            따라서, 상수의 출력 값이 1인 bias 노드를 생성하여 출력노드 Input으로 넣음



# 간단한 구조의 모델을 구성할 경우, 순차형(sequential)을 이용해서 직관적이고 빠르게 모델 구성
# 복잡한 구조의 모델을 구성할 경우, 함수형(functional) 모델 구성


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

""" 
loss :  [21.891637802124023, 3.2225496768951416]
RMSE :  4.678849926721399
R2 :  0.7291580664121273 """