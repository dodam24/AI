import numpy as np
# import sklearn as sk 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x)
print(x.shape)      # (20640, 8)   
print(y)
print(y.shape)      # (20640, 1)

print(dataset.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

print(dataset.DESCR)
""" .. _california_housing_dataset:

California Housing dataset
--------------------------

**Data Set Characteristics:**
 
    :Number of Instances: 20640

    :Number of Attributes: 8 numeric, predictive attributes and the target

    :Attribute Information:
        - MedInc        median income in block group
        - HouseAge      median house age in block group
        - AveRooms      average number of rooms per household
        - AveBedrms     average number of bedrooms per household
        - Population    block group population
        - AveOccup      average number of household members
        - Latitude      block group latitude
        - Longitude     block group longitude

    :Missing Attribute Values: None """

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(11))
model.add(Dense(65))
model.add(Dense(32))
model.add(Dense(48))
model.add(Dense(10))
model.add(Dense(75))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',     # adam: 방향과 학습률 두 가지를 모두 잡기 위한 방법
              metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', 
                              mode='min',
                              patience=10, 
                              restore_best_weights=True,
                              verbose=1)

hist = model.fit(x_train, y_train, epochs=300, batch_size=25, 
          validation_split=0.2, callbacks=[earlystopping], 
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

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

#5. 시각화
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
# plt.legend(loc='upper right')
plt.show()


""" Epoch 00029: early stopping
loss :  [0.9976897835731506, 0.8263436555862427] """