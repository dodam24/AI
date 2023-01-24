import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터 가져오기
# 데이터 경로
path = 'C:/study/_data/ddarung/'
path2 = 'C:/study_save'

# csv 파일 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

########## 결측치 처리 방법     1.삭제 ##########
# print(train_csv.isnull().sum())
train_csv = train_csv.dropna()    # 결측치 제거
x = train_csv.drop('count', axis=1)     # count 컬럼 삭제 (결과값에 해당하므로)
y = train_csv['count']      # count(결과)만 가져오기


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, train_size=0.7)

# scaler 설정 (데이터 전처리)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


""" #2. 모델링 (순차형)
model=Sequential()
model.add(Dense(50, input_dim=9))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation=linear))
model.add(Dense(1)) """

#2. 모델 구성 (함수형)          # Drop-out 적용 (overfitting 방지 목적)
input1 = Input(shape=(9,))
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

filepath='./_save/MCP/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5' #d:digit, f:float


# modelcheckpoint 설정
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      #filepath = path + 'MCP/keras30_ModelCheckPoint1.hdf5'
                      filepath = filepath + 'K31_ddarung_' + 'd_' + date + '_' + 'e_v_' + filename)


model.fit(x_train, y_train, epochs=1000, batch_size=15, validation_split=0.2, verbose=1, callbacks=[es, mcp])

model.save(path + 'keras31_dropout_save_model_ddarung.h5')   # 모델 저장

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



""" Epoch 00036: early stopping
mse :  2415.666259765625
mae :  37.76014709472656
==============================
R2 :  0.6645740939237724
RMSE :  49.14942527943336
============================== """