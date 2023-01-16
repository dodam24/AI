import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 준비
x = np.array([range(10), range(21, 31), range(201, 211)])   # 10개의 데이터가 리스트 형태로 묶임
# print(range(10))   # 0부터 10-1(=9)까지
print(x.shape)   # (3, 10) 3개의 덩어리 -> 전치
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])   # (2, 10) 2개의 덩어리 -> 전치
print(y.shape)

# [9, 3, 210] 넣었을 때, [10, 1.4] 나오는지 확인

x = x.T
print(x.shape)   # (10, 3)

y = y.T   
print(y.shape)  # (10, 2)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))   # 맨 처음 input과 output 값만 체크
model.add(Dense(70))
model.add(Dense(500))
model.add(Dense(30))
model.add(Dense(800)) 
model.add(Dense(50))
model.add(Dense(2)) 

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[9, 3, 210]])
print('[9, 3, 210]의 예측값 : ', result)

""" 결과
loss :  0.526901364326477
[9, 3, 210]의 예측값 :  [[-3.0245657  0.6614623]] 
 """
