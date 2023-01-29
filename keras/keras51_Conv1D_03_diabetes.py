import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)

# 스케일링
scaler= MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)      # (309, 10) (133, 10)

# Reshape로 형태 변환
x_train = x_train.reshape(309, 10, 1)
x_test = x_test.reshape(133, 10, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(10, 1), padding='same'))
model.add(Conv1D(32, 2, input_shape=(10, 1), padding='same'))
model.add(Conv1D(16, 2, input_shape=(10, 1), padding='same'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
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
                 validation_split=0.3, callbacks=[EarlyStopping], verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# print(hist.history['loss'])     # history의 loss값 불러오기

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)


""" 
Epoch 00097: early stopping
5/5 [==============================] - 0s 751us/step - loss: 3503.9858
loss :  3503.98583984375
RMSE :  59.19447469905232
R2 :  0.41076895610832265 """