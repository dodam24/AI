import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'   # . 은 현재 파일(study)을 의미. 데이터의 위치 표시
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# train_csv = pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)   # index_col=0 : 0번째 컬럼은 index임을 명시 (데이터 아님)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)      # (1459, 10) -> input_dim=10. but count(=y)에 해당하므로 count 분리. 따라서 input_dim=9
print(submission.shape)     # (715, 1)

print(train_csv.columns)   
""" Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'], dtype='object') """

print(train_csv.info())

""" 
결측치 처리 방법:
1. 결측치 있는 데이터 삭제 (null값)
2. 임의의 값 설정 (중간 값 or 0 입력) 
"""

print(test_csv.info())
print(train_csv.describe())

##### 결측치 처리 1. 제거 #####
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape)   # (1328, 10)

x = train_csv.drop(['count'], axis=1)   # count 컬럼 삭제 (컬럼 10개에서 9개로 변경됨)
print(x)                                # [1459 rows x 9 columns]
y = train_csv['count']                  # column(결과)만 추출 
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
model.add(Dense(1,input_dim=9))
model.add(Dense(15))
model.add(Dense(32))
model.add(Dense(67))
model.add(Dense(1))

#3. 컴파일, 훈련
import time 
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', 
                              mode='min',
                              patience=10, 
                              restore_best_weights=True,
                              verbose=1)
start = time.time()

hist = model.fit(x_train, y_train, epochs=400, batch_size=32, 
          validation_split=0.2, callbacks=[earlystopping],
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

# 데이터 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', 
         marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', 
         marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('boston loss')
plt.legend()
# plt.legeng(loc='upper right')
plt.show()

y_predict = model.predict(x_test)   # x_test로 y_predict 예측
print(y_predict)   # 결측치로 인해 nan값이 출력됨

# 결측치 수정

def RMSE(y_test, y_predict):   # RMSE 정의
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
submission['count'] = y_submit   # submission의 count열에 y_submit값 대입
print(submission)

submission.to_csv(path + 'submission_01091535.csv')   # to_csv에 '경로'와 '파일명' 입력


""" Epoch 00177: early stopping
loss :  [2913.16455078125, 2913.16455078125] """
