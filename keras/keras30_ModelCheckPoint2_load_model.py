import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = 'C:/study/_save/'

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (506, 13) (506,)

x_train, y_train, x_test, y_test = train_test_split(x, y, shuffle=True, random_state=333, train_size=0.7)

# Scaler 설정
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

""" 
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) """

model = load_model(path + 'MCP/keras30_modelCheckPoint1.hdf5')

""" 
#2. 모델 구성 (함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='linear')(dense1)
dense3 = Dense(30, activation='sigmoid')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='relu')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary() 


#3. 컴파일, 훈련 (생략) -> 위에 있는 load_model 사용
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# earlystopping 설정
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

# modelcheckpoint 설정
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath = path+'MCP/keras30_ModelCheckPoint1.hdf5')

model.fit(x_train, y_train, epochs=1000, batch_size=15, validation_split=0.2, verbose=1, callbacks=[es, mcp]) 
"""


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


# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.