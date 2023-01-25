import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler


#   그래프 한글 깨짐 방지
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


# 데이터 전처리 (스케일러 설정)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='linear')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1, activation='linear')(dense5)
model = Model(inputs=input1, outputs=output1)
model.summary()

path = 'C:/study/_save/' 
# path ='./_save/' 
# path = '../_save/'
model.save(path + 'keras29_1_save_model.h5')
# model.save('C:/study/_save/keras29_1_save_model.h5')


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# earlystopping 설정
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

# modelcheckpoint 설정
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath = path +'MCP/keras30_ModelCheckPoint1.hdf5')

model.fit(x_train, y_train, epochs=1000, batch_size=15, validation_split=0.2, verbose=1, callbacks=[es, mcp])


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


""" Epoch 00060: early stopping
mse : 17.18156623840332
mae : 2.679560661315918
==============================
R2 : 0.8248190526522153
RMSE : 4.145065377354448
============================== """