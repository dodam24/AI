import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
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


# scaler 설정
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)     # train 데이터로 fit, transfrom 적용하고 
x_test = scaler.transform(x_test)           # test 데이터는 transform만 적용
                                            # (x_train을 fit 적용한 가중치 값 범위에 맞춰서 x_test 데이터 변환)


#2. 모델 구성 (함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
drop1 = Dropout(0.5)(dense1)                            # Drop-out은 서로 연결된 연결망(layer)에서 0부터 1 사이의 확률로 뉴런을 제거(drop)하는 기법
dense2 = Dense(40, activation='sigmoid')(drop1)         # 어떤 특정한 설명변수 Feature만을 과도하게 집중 학습함으로써 발생할 수 있는 과대적합(Overfitting)을 방지하기 위한 목적으로 사용
drop2 = Dropout(0.3)(dense2)                            # Drop-out Rate는 하이퍼파라미터이며, 일반적으로 0.5로 설정 (뉴런 각각은 0.5의 확률로 제거될지 말지 랜덤하게 결정됨)    
dense3 = Dense(30, activation='linear')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(20, activation='relu')(drop3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1, activation='linear')(dense5)
model = Model(inputs=input1, outputs=output1)
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

model.fit(x_train, y_train, epochs=1000, batch_size=15, validation_split=0.2, verbose=1, callbacks=[es, mcp])

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



""" Epoch 00072: early stopping
mse :  21.4694766998291
mae :  3.058321237564087
==============================
R2: 0.7811001050978096
RMSE: 4.633516629287062
============================== """