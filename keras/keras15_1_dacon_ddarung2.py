import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
#1-1. 데이터 경로
path = './_data/ddarung/'   # . : 현재 파일(study)을 의미함 (데이터의 위치 표시)
#1-2. csv 파일 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    # index_col=0: 0번째 컬럼은 index임을 명시 (데이터 아님) 
# train_csv = pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)   
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)  # (1459, 10). count는 y값이므로 count 분리. 따라서 input_dim=9
print(submission.shape) # (715, 1)

print(train_csv.columns)   
""" 
Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
      dtype='object')
"""

print(train_csv.info())
""" 
Data columns (total 10 columns):
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    1459 non-null   int64
 1   hour_bef_temperature    1457 non-null   float64    # 결측치 2개 (1459개 기준)
 2   hour_bef_precipitation  1457 non-null   float64
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64
 9   count                   1459 non-null   float64
 """
 
""" 
결측치 처리 방법: 
1. 결측치 있는 데이터 삭제 (null값)
2. 임의의 값 설정 (중간 값, 0) 
"""

print(test_csv.info())      # non-null: 결측치 (데이터가 없는 것)
                            # 비어있는 값에 임의의 값을 넣어서 테스트 가능. 아예 데이터를 삭제하는 방법도 있음)
""" 
Data columns (total 9 columns):
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
 8   hour_bef_pm2.5          679 non-null    float64 
 """
# print(train_csv.describe())

x = train_csv.drop(['count'], axis=1)   # count 열 삭제
y = train_csv['count']                  # column(결과)만 추출

##### 결측치 처리 방법 1. 삭제 #####
print(train_csv.isnull().sum())     # null값의 개수 확인
""" 
None
hour                        0
hour_bef_temperature        2
hour_bef_precipitation      2
hour_bef_windspeed          9
hour_bef_humidity           2
hour_bef_visibility         2
hour_bef_ozone             76
hour_bef_pm10              90
hour_bef_pm2.5            117
count                       0 
"""

train_csv = train_csv.dropna()      # 결측값이 들어있는 행 전체 제거
print(train_csv.isnull().sum()) 
""" 
hour                      0
hour_bef_temperature      0
hour_bef_precipitation    0
hour_bef_windspeed        0
hour_bef_humidity         0
hour_bef_visibility       0
hour_bef_ozone            0
hour_bef_pm10             0
hour_bef_pm2.5            0
count                     0 
"""

print(train_csv.shape)      # (1328, 10)

x = train_csv.drop(['count'], axis=1)   # count 컬럼 삭제 (컬럼 10개에서 9개로 변경)
                                        # 데이터프레임의 열이나 행 삭제 작업 등을 할 때, axis(축) 지정
                                        # axis=0(index): 행 / axis=1(columns): 열
print(x)                                # [1459 rows x 9 columns]
y = train_csv['count']                  # count column(=결과)만 추출 
print(y)
print(y.shape)                          # (1459,)


# 빨간점 찍고 F5 누르면 중단점 실행
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=1234
)                                  

print(x_train.shape, x_test.shape)   # (929, 9) (399, 9)
print(y_train.shape, y_test.shape)   # (929,) (399,)

#2. 모델 구성
model=Sequential()
model.add(Dense(1, input_dim=9))
model.add(Dense(15))
model.add(Dense(32))
model.add(Dense(24))
model.add(Dense(1))

#3. 컴파일, 훈련
import time 
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)   # 그냥 돌리면 결측치로 인해 nan값이 출력됨 (에러)
print(y_predict)


# 결측치 수정
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)   # RMSE : 81.93167235968318

print("걸린 시간 : ", end - start)

# 제출할 파일
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape)   # (715, 1)

# .to_csv()를 사용해서
# submission_0105.csv를 완성시킬 것

print(submission)
submission['count'] = y_submit   # 비어있는 submission 파일의 count열에 y_submit 값 대입
print(submission)

submission.to_csv(path + 'submission_01050251.csv')   # to_csv에 '경로'와 '파일명' 입력

""" 
RMSE :  53.90369590007351

cpu 걸린 시간 : 63.03612923622131
gpu 걸린 시간 : 183.5358691215515 
"""