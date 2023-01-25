import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 준비
x = np.array([range(10)])
# print(range(10))   # 0부터 10-1(=9)까지
print(x.shape)   # (1, 10)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])
print(y.shape)   # (3, 10)

# [9] 넣었을 때, [10, 1.4] 나오는지 확인

x = x.T
print(x.shape)  # (10, 1)

y = y.T
print(y.shape)  # (10, 3) 

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=700, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x, y)     # 훈련시키지 않은 값으로 평가해야 정확한 예측이 가능
print('loss : ', loss)

result = model.predict([9])     # 행 무시, 열 우선! 이므로 예측 시, 열의 개수 동일하게 맞춰줄 것
print('[9]의 예측값 : ', result)


""" 결과
loss :  0.07102593034505844
[9]의 예측값 :  [[10.009682   1.6694055  0.1308834]] """