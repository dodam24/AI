import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

print(x.shape)  # (20,)
print(y.shape)  # (20,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', 
              metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)   # x_test와 y_test 비교해서 정확도 확인

print("========================")
print(y_test)
print(y_predict)
print("========================")

from sklearn.metrics import mean_squared_error  

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   # RMSE 정의 (np.sqrt: 제곱근)

print("RMSE : ", RMSE(y_test, y_predict))

# loss :  [14.923779487609863, 3.0428364276885986]    # 값이 2개인 이유: loss값, metrics값 