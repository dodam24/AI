import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score   # RMSE 만들기 위해 필요
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
path = './_data/ddarung/'   # . 은 현재 파일(study)을 의미. 데이터의 위치 표시
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# train_csv = pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)   # index_col=0 : 0번째 컬럼은 index임을 명시 (데이터 아님)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)   # (1459, 10) -> input_dim=10. but count(=y)에 해당하므로 count 분리. 따라서 input_dim=9
print(submission.shape)   # (715, 1)

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

""" 
결측치 처리 방법 
1. 결측치 있는 데이터 삭제 (null값)
2. 임의의 값 설정 (중간 값 or 0 입력) """

print(test_csv.info())
print(train_csv.describe())


##### 결측치 처리 1. 제거 #####
x = train_csv.drop(['count'], axis=1)       # count 컬럼 삭제 (컬럼 10개에서 9개로 변경됨)
print(x)                                    # [1459 rows x 9 columns]
y = train_csv['count']                      # column(결과)만 추출 
print(y)
print(y.shape)                              # (1459,)

print(train_csv.isnull().sum())             # null값의 개수 확인
train_csv = train_csv.dropna()              # 결측값이 들어있는 행 전체를 제거
print(train_csv.isnull().sum())             # 결측치 제거 후, null값위 개수 다시 확인
print(train_csv.shape)                      # (1328, 10)


# 빨간점 찍고 F5 누르면 중단점 실행
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=1234
)                                  

# 데이터 전처리
scaler = MinMaxScaler()                     # minmaxscaler 정의
# scaler = StandardScaler()
scaler.fit(x_train)                         # x값의 범위만큼의 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)           # x_train fit한 가중치 값 범위에 맞춰서 x_test 데이터 변환
test_csv = scaler.transform(test_csv)       # 제출 파일도 스케일링 해주어야 함.


print(x_train.shape, x_test.shape)      # (1021, 9) (438, 9)

x_train = x_train.reshape(1021, 9, 1)       
x_test = x_test.reshape(438, 9, 1)
print(x_train.shape, x_test.shape)



#2. 모델 구성 (순차형)
model = Sequential()
model.add(LSTM(64, input_shape=(9, 1)))
model.add(Dense(1, activation='linear'))

model.summary()

#3. 컴파일, 훈련
import time 
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',
                              patience=10, 
                              restore_best_weights=True,
                              verbose=1)
start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, 
          validation_split=0.2, callbacks=[EarlyStopping],
          verbose=2)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("==================================================")
print(hist) # <keras.callbacks.History object at 0x0000017442ACECA0>
print("==================================================")
print(hist.history)
print("==================================================")
print(hist.history['loss'])


y_predict = model.predict(x_test)   # x_test로 y_predict 예측
print(y_predict)                    # 결측치로 인해 nan값이 출력됨

# 결측치 수정

def RMSE(y_test, y_predict):   # RMSE 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)   # RMSE : 81.93167235968318

print("걸린 시간 : ", end - start)



# 제출할 파일
y_submit = model.predict(test_csv)      # 제출용 파일도 똑같이 스케일링 해주어야 함.
print(y_submit)
print(y_submit.shape)   # (715, 1)

# .to_csv()를 사용해서
# submission_0105.csv를 완성시킬 것

print(submission)
submission['count'] = y_submit      # submission의 count열에 y_submit 대입
print(submission)

submission.to_csv(path + 'submission_01111725.csv')     # to_csv에 경로와 파일명 입력