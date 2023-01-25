import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)


# scaler 설정
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성 (함수형)
input1 = Input(shape=(10,))
dense1 = Dense(50, activation='relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(40, activation='sigmoid')(drop1)
drop2 = Dropout(0.3)(dense2)
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
date = date.strftime("%m%d_%H%M")   # 0112_2313
print(date)
print(type(date))   # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # d:digit, f:float


# modelcheckpoint 설정
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      #filepath = path + 'MCP/keras30_ModelCheckPoint1.hdf5'
                      filepath = filepath + 'K31_diabetes_' + 'd_' + date + '_' + 'e_v_' + filename)


model.fit(x_train, y_train, epochs=1000, batch_size=15, validation_split=0.2, verbose=1, callbacks=[es, mcp])

model.save(path + 'keras31_dropout_save_model_diabetes.h5')

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



""" Epoch 00088: early stopping
mse :  2866.109130859375
mae :  43.56072235107422
==============================
R2 :  0.45941894308457387
RMSE :  53.53605612906221
============================== """