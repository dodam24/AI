import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array(range(1, 11))
y_train = np.array(range(1, 11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])

# x_train = x_train.T
# y_train = y_train.T

# 검증 데이터(validation) 추가
x_validation = np.array([14,15,16])   
y_validation = np.array([14,15,16])


#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=3,
          validation_data=(x_validation, y_validation))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)             # loss :  0.09585868567228317

result = model.predict([17])
print("17의 예측값 : ", result)     # 17의 예측값 :  [[16.259613]]