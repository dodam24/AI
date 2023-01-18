import numpy as np
import sklearn as sk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

# print(x)
print(x.shape)   # (442, 10)
# print(y)
print(y.shape)   # (442,)

print(dataset.feature_names)

print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123)

model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(49))
model.add(Dense(72))
model.add(Dense(15))
model.add(Dense(31))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=400, batch_size=16, 
          validation_split=0.3)

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
loss :  [3000.499267578125, 3000.499267578125]
RMSE :  54.7768124243929
R2 :  0.49543538502557816 
"""