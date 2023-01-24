import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (506, 13) (506,)

# print(np.min(x), np.max(x))     # 0.0 711.0

x_train, x_test, y_train, y_test= train_test_split(x, y, shuffle=True, random_state=333, test_size=0.2)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)   # 위의 두 줄과 동일한 코드
x_test = scaler.transform(x_test)

#2. 모델링 (함수형)
path = 'C:/study/_save/'
# model.save(path + 'keras29_1_save_model.h5')

model = load_model(path + 'keras29_1_save_model.h5')
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping    # earlystopping 추가

# earlystopping 설정

earlyStopping = EarlyStopping(
    monitor='val_loss',                 # history의 val_loss의 최소값을 이용
    mode='min',                         # accuracy 사용할 때는 정확도가 높을수록 좋으므로 mode='max'로 설정
    patience=15,                        
    restore_best_weights=True,
    verbose=1
)

model.fit(x_train, y_train, epochs=200, batch_size=1,
          validation_split=0.2, verbose=1, callbacks=[earlyStopping])
# val_loss를 기준으로 최소값이 n번(patience값 만큼) 이상 갱신 안 되면 훈련 중지

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse: ', mse)
print('mae: ', mae)

""" 
Epoch 00153: early stopping
mse:  99.3055648803711
mae:  6.912449836730957 """