import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

#1. 데이터
# 데이터 경로
path = 'C:/study/_data/bike/'

# CSV 파일 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

# 결측치 제거
# print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)          # (10886, 11)

# casual, registered 컬럼 제거, count 컬럼 제거
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

print(x.shape, y.shape)         # (10886, 8) (10886,)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.7, random_state=123)
print(x_train.shape, x_test.shape)      # (7620, 8) (3266, 8)

# train_csv 파일의 데이터 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(7620, 8, 1)
x_test = x_test.reshape(3266, 8, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(8, 1), padding='same'))
model.add(Conv1D(32, 2, input_shape=(8, 1), padding='same'))
model.add(Conv1D(16, 2, input_shape=(8, 1), padding='same'))
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
print(test_csv.shape)   # (6493, 8)
test_csv = test_csv.reshape(6493, 8, 1)

y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + 'sampleSubmission_0129_early_minmax_conv1d.csv')


""" 
Epoch 00091: early stopping
103/103 [==============================] - 0s 457us/step - loss: 23970.6016
loss :  23970.6015625
R2 :  0.26214828235396836 """