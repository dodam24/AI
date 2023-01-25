import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])    # (10, )
y = np.array(range(10))                 # (10, )
print(x)
print(y)

# 실습 : 넘파이 리스트 슬라이싱 7:3
x_train = x[:7]     # x[0:7] = x[:-3]
x_test = x[7:]      # x[7:10]
y_train = y[:7]     # y[:-3]       
y_test = y[7:]      # y[-3:]

print(x_train)
print(x_test)
print(y_train)
print(y_test)

""" print(x[:7], x[0:7], x[:-3])
print(x[7:], x[7:10])
print(y[:7], y[:-3])
print(y[7:], y[-3:]) """

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('[11]의 결과 : ', result)
