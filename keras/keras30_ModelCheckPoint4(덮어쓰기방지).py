import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 그래프 한글 깨짐 방지
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'   # 폰트가 저장된 경로에서 폰트 불러오기
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (506, 13) (506,)
print(np.min(x), np.max(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)


# scaler 설정
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성 (함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50,activation='relu')(input1)
dense2 = Dense(40,activation='sigmoid')(dense1)
dense3 = Dense(30,activation='linear')(dense2)
dense4 = Dense(20,activation='relu')(dense3)
dense5 = Dense(10,activation='relu')(dense4)
output1 = Dense(1,activation='linear')(dense5)
model = Model(inputs=input1, outputs=output1)

model.summary()     # Total params: 4,811

path = 'C:/study/_save/' 
# path ='./_save/' 
# path = '../_save/'
model.save(path+'keras29_1_save_model.h5')
# model.save('C:/study/_save/keras29_1_save_model.h5')


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# earlystopping 설정
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)


# 파일 이름 설정 (덮어쓰기 방지)
import datetime

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")     # 0112_2313
print(date)
print(type(date))   # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# modelcheckpoint 설정
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath = filepath+'k30_'+date+'_'+filename)

model.fit(x_train, y_train, epochs=1000, batch_size=15, validation_split=0.2, verbose=1, callbacks=[es, mcp])

model.save(path+'keras30_ModelCheckpoint3_save_model.h5')


#4. 평가, 예측
print("=============== 1. 기본 출력 ===============")
mse, mae = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('mse : ', mse)
print('r2 : ', r2)

print("=============== 2. load_model 출력 ===============")
model2 = load_model(path + 'keras30_ModelCheckPoint3_save_model.h5')

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('mse : ', mse)
print('r2 : ', r2)

print("=============== 3. ModelCheckPoint 출력 ===============")
model3 = load_model(path + 'MCP/keras30_ModelCheckPoint3.hdf5')
mse, mae = model.evaluate(x_test, y_test)

r2 = r2_score(y_test, y_predict)

print('mse : ', mse)
print('r2 : ', r2)
