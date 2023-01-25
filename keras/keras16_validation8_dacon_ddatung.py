# [실습]
# 1. train 0.7 이상
# 2. R2 : 0.8 이상 / RMSE 사용
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sklearn as sk
print(sk.__version__)   # 1.1.3


#1. 데이터
path = './_data/ddarung/'

# CSV 파일 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    # 첫 번째 열인 'ID' 변수를 Index로 지정 
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)      # (1459, 10). 맨 끝의 count는 결과(y)값이므로 분리. 따라서 input_dim=9 
print(train_csv.columns)
""" Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'], dtype='object') """
      
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
      
print(train_csv.describe()) 


# 결측치 처리 방법: 
x = train_csv.drop(['count'], axis=1)   # count 컬럼 삭제
# print(x)
y = train_csv['count']                  # count(결과)만 추출
# print(y)   
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
# print(y.shape)    # (1459,)

print(train_csv.isnull().sum())     # null값의 개수 확인
train_csv = train_csv.dropna()      # 결측값이 들어있는 행 전체를 제거
print(train_csv.isnull().sum())     # 결측치 제거 후, null값 개수 다시 확인
print(train_csv.shape)              # (1328, 10) 
                                    # 결측치 제거 전에는 train_csv.shape = (1459, 10)

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

      
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=123
)

print(x_train.shape, x_test.shape)      # (929, 9) (399, 9)
print(y_train.shape, y_test.shape)      # (929,) (399,)
                   

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=9))
model.add(Dense(55))
model.add(Dense(20))
model.add(Dense(35))
model.add(Dense(70))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
            metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   # RMSE: MSE의 제곱근 (np.sqrt)
print("RMSE : ", RMSE(y_test, y_predict))

""" def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) 
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse) """


r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 제출할 파일
y_submit = model.predict(test_csv)

""" ValueError: Input contains NaN.
모델을 그냥 돌리면 Nan값 때문에 에러 발생. 따라서 결측치 수정 또는 제거 해야 함. """


""" loss :  [2938.183349609375, 39.18789291381836]
RMSE :  54.205006639222965
R2 :  0.5563860413190791 """