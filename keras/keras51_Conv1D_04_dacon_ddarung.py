import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler


#1. 데이터

# 데이터 경로
path = 'C:/study/_data/ddarung/'

# CSV 파일 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

# train_csv 데이터의 결측치 삭제
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()      # 결측치 제거

# train 데이터에서 count 컬럼 값만 가져오기
x = train_csv.drop('count', axis=1)     # count 컬럼 제거
y = train_csv['count']                  # count 컬럼 값만 가져오기

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=123)
print(x_train.shape, x_test.shape)      # (929, 9) (399, 9)

# train_csv 파일의 데이터 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(929, 9, 1)
x_test = x_test.reshape(399, 9, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(9, 1), padding='same'))
model.add(Conv1D(32, 2, input_shape=(9, 1), padding='same'))
model.add(Conv1D(16, 2, input_shape=(9, 1), padding='same'))
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

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(RMSE(y_test, y_predict))
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)


# 제출용 파일 (위에서 데이터 스케일링 했으므로 제출용 데이터도 스케일링 해야 함)
test_csv = scaler.transform(test_csv)
print(test_csv.shape)   # (715, 9)
test_csv = test_csv.reshape(715, 9, 1)

y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + 'sampleSubmission_0129_early_minmax_conv1d.csv')


""" 
Epoch 00064: early stopping
13/13 [==============================] - 0s 500us/step - loss: 2732.1116
loss :  2732.111572265625
R2 :  0.5874991792009168 """