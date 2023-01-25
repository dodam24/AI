import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])                # 데이터셋 3개

y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape)   # (3, 10)
print(y.shape)   # (10,)

x = x.T
print(x.shape)   # (10, 3)

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=3))    # input data set이 3차 행렬이므로 input_dim에 3 대입
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=3, batch_size=6)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[10, 1.4, 20]])
print('[10, 1.4, 20]의 예측값 : ', result)

# 예측값과 loss의 최적값이 서로 다를 때: loss값이 우선이므로 loss 값이 더 좋은 것으로 선택

'''
결과 : epochs=300, batch_size=6, loss=0.007648753933608532
'''