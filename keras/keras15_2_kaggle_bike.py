import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)   # (10886, 11)   # count=y(결과)에 해당하므로 input_dim=10
print(sampleSubmission.shape)   # (6493, 1)

print(train_csv.columns)
#Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object').

# print(train_csv.info())
# print(test_csv.info())
# print(train_csv.describe())

# *** x는 casual, registered, count 빼고 y는 train에서 count만 가져올 것!
train_csv=train_csv.drop(['casual', 'registered'], axis=1)
x = train_csv.drop(['count'], axis=1)
print(x)   # [10886 rows x 10 columns]
y = train_csv['count']   # column(결과)만 추출
print(y)   # [10886 rows x 10 columns]
print(y.shape)   # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=123
)   

print(x_train.shape, x_test.shape)   # (7620, 10) (3266, 10)
print(y_train.shape, y_test.shape)   # (7620,) (3266,)


#2. 모델 구성
model=Sequential()
model.add(Dense(4,input_dim=8, activation='relu'))   # activation의 defalut값 = linear
model.add(Dense(1, activation='relu'))
model.add(Dense(28))
model.add(Dense(3, activation='relu'))
model.add(Dense(42))
model.add(Dense(5))
model.add(Dense(7, activation='relu'))
model.add(Dense(65, activation='relu'))
model.add(Dense(33))
model.add(Dense(6, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
import time 
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x_train, y_train, epochs=1515, batch_size=15)
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  # x_test로 y_predict 예측
print(y_predict)

def RMSE(y_test, y_predict):   # RMSE 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)   # RMSE :

# 제출할 파일 생성
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)   #

print(sampleSubmission)
sampleSubmission['count'] = y_submit   # submission의 count열에 y_submit 대입
print(sampleSubmission)

sampleSubmission.to_csv(path + 'sampleSubmission_01061225.csv')   # to_csv에 경로와 파일명 입력