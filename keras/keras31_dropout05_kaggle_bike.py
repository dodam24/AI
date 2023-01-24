import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터 가져오기
# 데이터 경로
path = 'C:/study/_data/bike/'
path2 = 'C:/study_save'

# csv 파일 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

########## 결측치 처리: 삭제 ##########
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape)          # (10886, 11)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

print(x.shape, y.shape)         # (10886, 8) (10886)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, train_size=0.7)

# scaler 설정
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print('x : ', x_train.shape, x_test.shape)      # x: (7620, 8) (3266, 8)
print('y : ', y_train.shape,y_test.shape)       # y: (7620,) (3266,)


""" #2. 모델링 (순차형)
model = Sequential()
model.add(Dense(50, input_dim=9))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation=linear))
model.add(Dense(1)) """

#2.모델 구성 (함수형)
input1 = Input(shape=(8,))
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
# path = './_save/' 
# path = '../_save/'
# model.save(path + 'keras29_1_save_model.h5')
# model.save('C:/study/_save/keras29_1_save_model.h5')


#3.컴파일, 훈련
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
                      filepath = filepath + 'K31_kaggle_' + 'd_' + date + '_' + 'e_v_' + filename)


model.fit(x_train, y_train, epochs=1000, batch_size=15, validation_split=0.2, verbose=1, callbacks=[es, mcp])

model.save(path + 'keras31_dropout_save_model_kaggle.h5')

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
mse : 22778.740234375
mae : 111.89749145507812
==============================
R2 : 0.29518252320884175
RMSE : 150.92628681159042
============================== """