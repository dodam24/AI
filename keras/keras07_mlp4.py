import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 준비
x = np.array([range(10)])   # (10,) (10, 1)
# print(range(10))   # 0부터 10-1(=9)까지
print(x.shape)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])   # (3, 10)
print(y.shape)

# [9] 넣었을 때, [10, 1.4] 나오는지 확인

x = x.T
print(x.shape)   # (10, 1)

y = y.T   # (10, 3) 
print(y.shape)

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(3))

model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=700, batch_size=4)

loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([9])
print('[9]의 예측값 : ', result)