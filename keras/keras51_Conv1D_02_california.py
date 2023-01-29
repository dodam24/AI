import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)

print(x_train.shape, x_test.shape)      # (14447, 8) (6193, 8)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)

x_train = x_train.reshape(14447, 8, 1)
x_test = x_test.reshape(6193, 8, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(8, 1), padding='same'))
model.add(Conv1D(64, 2, input_shape=(8, 1), padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=10, 
                              restore_best_weights=True, 
                              verbose=1)
hist = model.fit(x_train, y_train, epochs=100, batch_size=25,
                 validation_split=0.3, callbacks=[EarlyStopping], verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print(hist.history['val_loss'])

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


""" 
Epoch 00055: early stopping
194/194 [==============================] - 0s 460us/step - loss: 0.3942 - mae: 0.4407
RMSE :  0.6278355767901905
R2 :  0.7018976298290402 """