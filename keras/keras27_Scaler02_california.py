import numpy as np
import sklearn as sk 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x)
print(x.shape)   # (20640, 8)   
print(y)
print(y.shape)  # (20640, 1)

print(dataset.feature_names)

print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123)

scaler = MinMaxScaler() # minmaxscaler 정의
#  scaler = StandardScaler()
scaler.fit(x_train) # x값의 범위만큼의 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)# 시작 (transform해야 바뀐다.)

#2. 모델 구성
model=Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(75))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',
                              patience=10, 
                              restore_best_weights=True,
                              verbose=1)
hist = model.fit(x_train, y_train, epochs=300, batch_size=25, 
          validation_split=0.2, callbacks=[EarlyStopping], 
          verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


print("==================================================")
print(hist) # <keras.callbacks.History object at 0x0000017442ACECA0>
print("==================================================")
print(hist.history)
print("==================================================")
print(hist.history['loss'])

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

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결과
# Epoch 00019: early stopping
