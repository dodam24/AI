import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2
)

print(x_train.shape, x_test.shape)      # (120, 4) (30, 4)

x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(4, 1), padding='same'))
model.add(Conv1D(64, 2, input_shape=(4, 1), padding='same'))
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


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)



""" Epoch 00045: early stopping
1/1 [==============================] - 0s 10ms/step - loss: 0.0508
loss :  0.05083121359348297
RMSE :  0.22545778839112635
R2 :  0.9361060154865977 """