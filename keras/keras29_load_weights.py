import numpy as np

from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=333)

# Scaler 설정
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델링 (함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='linear')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1, activation='linear')(dense5)

model = Model(inputs=input1, outputs=output1)
model.summary()

# model.save_weights(path + 'keras29_5_save_weights1.h5')
# model.load_weights(path + 'keras29_5_save_weights1.h5')
# Model이 저장되지 않고 가중치만 저장됨 (사용할 경우, 모델 설정이 필요함)
# weights를 가져왔으므로 fit은 필요없음
# RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# EarlyStopping 설정
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)

hist = model.fit(x_train, y_train,
                 epochs=500,
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)


path = 'C:/study/_save/'

# model.save_weights(path + 'keras29_5_save_weights2.h5')   
model.load_weights(path + 'keras29_5_save_weights2.h5')
# RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.
# weights만 저장되었으므로 compile 필요

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


""" RMSE :  4.2720104782711115
R2 :  0.813924703562832 """


""" 
save_model, load_model: 모델 전체를 파일로 저장하고 불러오는 방법
save_weights, load_weights: 가중치만 파일로 저장하고 불러오는 방법

save & load_model과 save & load_weights의 차이점:
    model save 및 load는 모델 전체를 저장하기 때문에, load 이후에 별도로 처리할 필요가 없어 매우 간편함
    weights save 및 load는 가중치만 저장하기 때문에, 모델 architecture를 동일하게 만들어야 함
    
    ex. save로 저장한 파일은 79K
        save_weights로 저장한 파일은 40K
        따라서 weights 파일이 훨씬 더 작기 때문에 성능이 향상됨 (시간 감소 등) """