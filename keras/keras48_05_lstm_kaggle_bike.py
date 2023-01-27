import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)


train_csv=train_csv.drop(['casual', 'registered'], axis=1)
x = train_csv.drop(['count'], axis=1)   
y = train_csv['count']                  
x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.3, shuffle=True, random_state=123)  


# 데이터 전처리 (스케일링)
scaler = MinMaxScaler()     
# scaler = StandardScaler()

scaler.fit(x_train)         
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)        

test_csv = scaler.transform(test_csv) 


x_train = x_train.reshape(3265, 8, 1)       
x_test = x_test.reshape(7621, 8, 1)
print(x_train.shape, x_test.shape)


#2. 모델 구성 (순차형)
model = Sequential()
model.add(LSTM(64, input_shape=(8, 1)))
model.add(Dense(1, activation='linear'))

model.summary()


#3. 컴파일, 훈련
import time 
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',
                              patience=10, 
                              restore_best_weights=True,
                              verbose=1)
start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, 
          validation_split=0.2, callbacks=[EarlyStopping], 
          verbose=3)
end = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("==================================================")
print(hist) # <keras.callbacks.History object at 0x0000017442ACECA0>
print("==================================================")
print(hist.history)
print("==================================================")
print(hist.history['loss'])


y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)



# 제출할 파일 생성
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) 

print(sampleSubmission)
sampleSubmission['count'] = y_submit   # submission의 count열에 y_submit값 대입
print(sampleSubmission)

sampleSubmission.to_csv(path + 'sampleSubmission_01111725.csv')   # to_csv에 '경로'와 '파일명' 입력