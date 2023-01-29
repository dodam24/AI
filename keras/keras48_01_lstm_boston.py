import numpy as np
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)         # (506, 13) (506,)
print(np.min(x), np.max(x))     # 0.0 711.0


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)
print(x_train.shape, x_test.shape)      # (404, 13) (102, 13)

x_train = x_train.reshape(404, 13, 1)       
x_test = x_test.reshape(102, 13, 1)
print(x_train.shape, x_test.shape)

''' # scaler 설정
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)     # train 데이터로 fit, transfrom 적용하고 
x_test = scaler.transform(x_test)           # test 데이터는 transform만 적용
                                            # (x_train을 fit 적용한 가중치 값 범위에 맞춰서 x_test 데이터 변환)
 '''
#2. 모델 구성 (순차형)
model = Sequential()
model.add(LSTM(64, input_shape=(13, 1)))
model.add(Dense(50, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

model.summary()

path = 'C:/study/_save/'    # 경로 설정
# path ='./_save/' 
# path = '../_save/'

model.save(path + 'keras31_dropout01_boston.h5')                # 모델 저장
# model.save('C:/study/_save/keras31_dropout01_boston.h5')


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# earlystopping 설정
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)


# 파일 이름 설정 (덮어쓰기 방지)
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")     # 0112_2313
print(date)
print(type(date))     # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # d:digit, f:float


# modelcheckpoint 설정
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      # filepath = path + 'MCP/keras31_dropout_ModelCheckPoint_boston.hdf5'
                      filepath = filepath + 'K31_dropout_boston_' + 'd_' + date + '_' + 'e_v_' + filename)

model.fit(x_train, y_train, epochs=10, batch_size=15, validation_split=0.2, verbose=1, callbacks=[es, mcp])

model.save(path + 'keras31_dropout_save_model_boston.h5')

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

print("==============================")
print("R2 : ", r2)
print("RMSE : ", RMSE(y_test, y_predict))
print("==============================")