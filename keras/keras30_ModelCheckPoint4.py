import numpy as np
# import pandas as pd
# # import sklearn as sk
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = './_save/'
# path = '../_save/'
# path = 'c:/study/_save/'


#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.3, shuffle=True, random_state=123)


scaler = MinMaxScaler() # minmaxscaler 정의
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)# 시작 (transform해야 바뀐다.)


#2. 모델 구성(함수형) # 순차형과 반대로 레이어 구성
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='linear')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(10, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1) # 순차형과 달리 model 형태를 마지막에 정의
model.summary()

# model.save(path + 'keras29_1_save_model.h5')



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
            metrics=['mae'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=False)


################## 파일이 생성된 시간과 성능을 파일 이름에 넣기 ##################
import datetime # datetime 이라는 데이터 형식
date = datetime.datetime.now()
print(date) # 2023-01-12 14:57:49.184737
print(type(date)) # 파일명에 사용할 예정이므로 문자열 형태로 바꿔줘야 함
date = date.strftime("%m%d_%H%M") # 월 일 _시 분 (0112_1457)
print(date) # 0112_1502
print(type(date))

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # 정수로 4자리까지 받아들임. 4f(소수 넷째자리) # 0037-0.0048.hdf5
# 에포크 4번째 loss 소수 4번째 자리까지

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                    #   filepath=path + "keras30_ModelCheckPoint4.hdf5") # MCP 파일은 가중치가 저장되어 있는 파일
                      filepath=path + 'K30_' + date + '_' + filename)
######################################################################################


model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)

model.save(path + 'keras3_ModelCheckPoint3_save_model.h5')


# model.save(path + 'keras29_3_save_model.h5') # 훈련 시키고 나서 모델 세이브 (모델과 가중치 모두 저장)
# 0.7729 ~

model = load_model(path + "keras30_ModelCheckPoint1.hdf5")



#4. 평가, 예측   ###### model이랑 model2 비교
model2 = load_model(path + "keras30_ModelCheckPoint3.h5")

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

loss = model2.evaluate(x_test, y_test)

y_predict = model2.predict(x_test)  # x_test로 y_predict 예측
# print("===========")
# print(y_test)
# print(y_predict)
# print("==========")
print("load_model 출력")
print("loss : ", loss)
print("RMSE : ", RMSE(y_test, y_predict))
print("R2 : ", r2_score(y_test, y_predict))