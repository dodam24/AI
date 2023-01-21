import numpy as np
import sklearn as sk

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x)
print(x.shape)  # (442, 10)
# print(y)
print(y.shape)  # (442,)

print(datasets.feature_names)

print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',
                              patience=10, 
                              restore_best_weights=True,
                              verbose=1)

hist = model.fit(x_train, y_train, epochs=200, batch_size=25, 
          validation_split=0.2, callbacks=[EarlyStopping], 
          verbose=1)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("==================================================")
print(hist) # <keras.callbacks.History object at 0x0000017442ACECA0>
print("==================================================")
print(hist.history)
print("==================================================")
print(hist.history['loss'])

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 시각화
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


""" Epoch 00058: early stopping
loss :  3139.808837890625 """