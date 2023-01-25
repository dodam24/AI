import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 

#1. 데이터 경로
path = './_data/ddarung/'   # . : 현재 파일(study)을 의미함 (데이터의 위치 표시)

# CSV 파일 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# train_csv = pd.read_csv('./_data/ddaru7ng/train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)   # (1459, 10). count는 y값이므로 count 분리. 따라서 input_dim=9
print(train_csv.columns)
""" Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
      dtype='object') """
      
print(train_csv.info())
""" Data columns (total 10 columns):
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    1459 non-null   int64
 1   hour_bef_temperature    1457 non-null   float64
 2   hour_bef_precipitation  1457 non-null   float64        # 결측치 2개 (1459개 기준)
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64
 9   count                   1459 non-null   float64 """
 
"""  결측치 처리 방법:
     1. 결측치 있는 데이터 삭제 (null값)
     2. 임의의 값 설정 (중간 값, 0) """
   
     
print(test_csv.info())
""" Data columns (total 9 columns):
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    715 non-null    int64
 1   hour_bef_temperature    714 non-null    float64
 2   hour_bef_precipitation  714 non-null    float64
 3   hour_bef_windspeed      714 non-null    float64
 4   hour_bef_humidity       714 non-null    float64
 5   hour_bef_visibility     714 non-null    float64
 6   hour_bef_ozone          680 non-null    float64
 7   hour_bef_pm10           678 non-null    float64
 8   hour_bef_pm2.5          679 non-null    float64 """


# print(train_csv.describe())
""" hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  hour_bef_humidity  hour_bef_visibility  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5        count
count  1459.000000           1457.000000             1457.000000         1450.000000        1457.000000          1457.000000     1383.000000    1369.000000     1342.000000  1459.000000
mean     11.493489             16.717433                0.031572            2.479034          52.231297          1405.216884        0.039149      57.168736       30.327124   108.563400
std       6.922790              5.239150                0.174917            1.378265          20.370387           583.131708        0.019509      31.771019       14.713252    82.631733
min       0.000000              3.100000                0.000000            0.000000           7.000000            78.000000        0.003000       9.000000        8.000000     1.000000
25%       5.500000             12.800000                0.000000            1.400000          36.000000           879.000000        0.025500      36.000000       20.000000    37.000000
50%      11.000000             16.600000                0.000000            2.300000          51.000000          1577.000000        0.039000      51.000000       26.000000    96.000000
75%      17.500000             20.100000                0.000000            3.400000          69.000000          1994.000000        0.052000      69.000000       37.000000   150.000000
max      23.000000             30.000000                1.000000            8.000000          99.000000          2000.000000        0.125000     269.000000       90.000000   431.000000 """

x = train_csv.drop(['count'], axis=1)   # count 컬럼 삭제 (컬럼 개수 10개에서 9개로 변경)
print(x)                                # [1459 rows x 9 columns]

y = train_csv['count']                  # column(결과)만 추출
print(y)
""" id
3        49.0
6       159.0
7        26.0
8        57.0
9       431.0
        ...
2174     21.0
2175     20.0
2176     22.0
2178    216.0
2179    170.0
Name: count, Length: 1459 """

print(y.shape)                          # (1459,)


# 결측치 처리 방법: 
# 모델 돌리기 전에 결측치 삭제
print(train_csv.isnull().sum())     # null값의 개수 확인
train_csv = train_csv.dropna()      # 결측값이 들어있는 행 전체를 제거
print(train_csv.isnull().sum())     # 결측치 제거 후, null값의 개수 다시 확인
print(train_csv.shape)              # (1328, 10) 
                                    # 결측치 제거 전에는 train_csv.shape = (1459, 10)

x = train_csv.drop(['count'], axis=1)   # train 데이터의 count 컬럼 삭제
y = train_csv['count']                  # count(결과)만 추출


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=1234
)

print(x_train.shape, x_test.shape)  # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape)  # (1021,) (438,)

#2. 모델 구성
model=Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)   # x_test로 y_predict 예측
print(y_predict)                    # 그냥 돌리면 결측치 때문에 nan값 에러 발생

# 결측치 해결
# keras15_dacon_ddarung2.py 참고

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# 제출할 파일
y_submit = model.predict(test_csv)      # test_csv의 값을 예측해서 y_submit에 대입
submission['count'] = y_submit          # y_submit의 값을 submission 파일의 count에 대입
print(submission)

submission.to_csv(path + 'submission_01212005.csv')     # to_csv에 '경로'와 '파일명' 입력


# RMSE : 77.8888497835806