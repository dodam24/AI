import numpy as np
import sklearn as sk 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x)
print(x.shape)   # (20640, 8)   
print(y)
print(y.shape)   # (20640, 1)

print(dataset.feature_names)

print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123)

print(x_train.shape, x_test.shape)      # (14447, 8) (6193, 8)

x_train = x_train.reshape(14447, 2, 2, 2)       
x_test = x_test.reshape(6193, 2, 2, 2)
print(x_train.shape, x_test.shape)


''' scaler = MinMaxScaler()                     # minmaxscaler 정의
# scaler = StandardScaler()
scaler.fit(x_train)                             # x값의 범위만큼 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)               # x_train fit한 가중치 값 범위에 맞춰서 x_test 데이터 변환
 '''
 
#2. 모델 구성 (순차형)
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='linear'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',
                              patience=10, 
                              restore_best_weights=True,
                              verbose=1)
hist = model.fit(x_train, y_train, epochs=10, batch_size=25,   
          validation_split=0.2, callbacks=[EarlyStopping], 
          verbose=1)

""" 
신경망 모델의 훈련에서 사용되는 fit() 메서드는 History 객체를 반환
History.history 속성: 훈련 과정에서 에포크(epoch)에 따른 정확도(accuracy)와 같은 지표와 손실값 기록
                    또한, 검증(validation)의 지표와 손실값도 기록 """
                   
                    
""" 
validation_split: 0에서 1 사이의 값으로,
                0.25로 설정해주면 훈련(train) 데이터의 25%를 검증에 사용
                if 60,000개의 데이터 중 45,000개를 훈련에 사용하고 15,000개를 검증에 사용 
 """

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


print("==================================================")
print(hist)     # <keras.callbacks.History object at 0x0000017442ACECA0>
print("==================================================")
print(hist.history)     # loss의 변화량 값을 dictionary 형태로 출력. (key:values) 즉, 키-값의 쌍형태. value: 리스트형     
print("==================================================")
print(hist.history['loss'])     # loss값만 출력


y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


""" RMSE :  1.1070910012490482
R2 :  0.07308432381509733 """