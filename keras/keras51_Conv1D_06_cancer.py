import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=123
)

print(x_train.shape, x_test.shape)      # (455, 30) (114, 30)

x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(30, 1), padding='same'))
model.add(Conv1D(64, 2, input_shape=(30, 1), padding='same'))
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



""" Epoch 00042: early stopping
4/4 [==============================] - 0s 667us/step - loss: 0.1177
loss :  0.11769326776266098
RMSE :  0.3430644668518805
R2 :  0.48896050902135635 """