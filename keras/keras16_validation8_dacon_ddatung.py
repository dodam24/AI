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
train_csv = pd.read_csv(path + 'train.csv', index_col=0)    # 첫 번째 열인 '~' 변수를 Index로 지정 
test_csv = pd.read_csv(path + 'test.csv')
submission = pd.read_csv



#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(55))
model.add(Dense(20))
model.add(Dense(35))
model.add(Dense(70))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
            metrics=['mae'])
model.fit(x_train, y_train, epochs=4000, batch_size=32,
          validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   # RMSE: MSE의 제곱근 (np.sqrt)
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 제출할 파일
y_submit = model.predcit(test_csv)

