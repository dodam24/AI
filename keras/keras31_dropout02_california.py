import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)


# scaler 설정
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성 (함수형)
input1 = Input(shape=(8,))
dense1 = Dense(50, activation='relu')(input1)
drop1 = Dropout(0.5)(dense1)                            # Drop-out: 서로 연결된 연결망(layer)에서 0부터 1 사이의 확률로 뉴런을 제거(drop)하는 기법
dense2 = Dense(40, activation='sigmoid')(drop1)         # 어떤 특정한 설명변수 Feature만을 과도하게 집중 학습함으로써 발생할 수 있는 과대적합(Overfitting)을 방지하기 위한 목적으로 사용
drop2 = Dropout(0.3)(dense2)                            # Drop-out Rate는 하이퍼파라미터이며, 일반적으로 0.5로 설정 (뉴런 각각은 0.5의 확률로 제거될지 말지 랜덤하게 결정됨)
dense3 = Dense(30, activation='linear')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(20, activation='relu')(drop3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1, activation='linear')(dense5)
model = Model(inputs=input1, outputs=output1)
model.summary()

path = 'C:/study/_save/' 
# path ='./_save/' 
# path = '../_save/'
# model.save(path+'keras29_1_save_model.h5')
# model.save('C:/study/_save/keras29_1_save_model.h5')


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# earlystopping 설정
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)


# 파일 이름 설정 (덮어쓰기 방지)
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   # strftime(): 원하는 서식을 지정해 날짜 형식 변경    # 0112_2313
print(date)
print(type(date))   # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # d:digit, f:float


# modelcheckpoint 설정
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      # filepath = path + 'MCP/keras30_ModelCheckPoint1.hdf5'
                      filepath = filepath + 'K31_cali_' + 'd_' + date + '_' + 'e_v_' + filename)


model.fit(x_train, y_train, epochs=1000, batch_size=15, validation_split=0.2, verbose=1, callbacks=[es, mcp])

model.save(path + 'keras31_dropout_save_model_california.h5')

#4.평가, 예측
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



""" Epoch 00067: early stopping
mse :  0.4092679023742676
mae :  0.45726174116134644
==============================
R2 :  0.678773352382919
RMSE :  0.639740433121515
============================== """