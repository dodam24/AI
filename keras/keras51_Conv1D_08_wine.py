import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (178, 13) (178,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
)

print(x_train.shape, x_test.shape)      # (142, 13) (36, 13)

x_train = x_train.reshape(142, 13, 1)
x_test = x_test.reshape(36, 13, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(13, 1), padding='same'))
model.add(Conv1D(64, 2, input_shape=(13, 1), padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=10,
                              restore_best_weights=True,
                              verbose=1)
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=25,
                 validation_split=0.2, callbacks=[EarlyStopping], verbose=1)


#4. #4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)



""" Epoch 00071: early stopping
2/2 [==============================] - 0s 1ms/step - loss: 0.1334
loss :  0.13343574106693268
RMSE :  0.36528837462940095
R2 :  0.7888491657574495 """