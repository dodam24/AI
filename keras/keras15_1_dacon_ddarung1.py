import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score   # RMSE 만들기 위해 필요

#1. 데이터
path = './_data/ddarung/'   # . 은 현재 파일(study)을 의미. 데이터의 위치 표시
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# train_csv = pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)   # index_col=0 : 0번째 컬럼은 index임을 명시 (데이터 아님)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)   # (1459, 10) -> input_dim=10. but count(=y)에 해당하므로 count 분리. 따라서 input_dim=9

print(train_csv.columns)   
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#      dtype='object')

print(train_csv.info())
# #   Column                  Non-Null Count  Dtype
#---  ------                  --------------  -----
# 0   hour                    1459 non-null   int64
# 1   hour_bef_temperature    1457 non-null   float64   # 결측치 2개 (1459개 기준)
# 2   hour_bef_precipitation  1457 non-null   float64

# # 결측치 처리 방법 
# 1. 결측치 있는 데이터 삭제 (null값)
# 2. 임의의 값 설정 (중간 값 or 0의 값 입력)

print(test_csv.info())
print(train_csv.describe())

x = train_csv.drop(['count'], axis=1)   # count 컬럼 삭제 (컬럼 10개에서 9개로 변경됨)
print(x)   # [1459 rows x 9 columns]
y = train_csv['count']   # column(결과)만 추출 
print(y)
print(y.shape)   # (1459,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=1234
)                                  

print(x_train.shape, x_test.shape)   # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape)   # (1021,) (438,)

#2. 모델 구성
model=Sequential()
model.add(Dense(1,input_dim=9))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)   # x_test로 y_predict 예측(?)
print(y_predict)   # 결측치로 인해 nan값이 출력됨

# 결측치 해결
# keras15_dacon_ddarung2.py 참고

def RMSE(y_test, y_predict):   # RMSE 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# 제출할 파일
y_submit = model.predict(test_csv)