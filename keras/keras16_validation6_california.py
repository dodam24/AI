import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

# print(x)
print(x.shape)   # (20640, 8)
# print(y)
print(y.shape)   # (20640,)

print(dataset.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(15))
model.add(Dense(25))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
            metrics=['mae'])
model.fit(x_train, y_train, epochs=500, batch_size=32,
          validation_split=0.3)     # 전체 데이터의 마지막 30%를 검증 데이터로 사용

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
loss :  [0.6636654138565063, 0.5617427825927734]
RMSE :  0.8146567675923694
R2 :  0.498093378745457
 """