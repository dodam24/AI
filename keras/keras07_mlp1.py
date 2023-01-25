import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape)  # (2, 10)   data.shape : 행렬의 구조
                # (10, 2) 형태로 변환  ex. [[1,1], [2,1], ...]의 형태
print(y.shape)  # (10,)

# (2, 10) 데이터를 (10, 2) 데이터로 변환
x = x.T   # 행과 열 전치
print(x.shape)   # (10, 2)

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=2))    # input: 2개, output: 5개, input_dim: 열의 개수
                                    # 행 무시, 열 우선
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=4)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[10, 1.4]])
print('[10, 1.4]의 예측값 : ', result)

'''
결과 : epochs=200, batch_size=4, loss=0.00522084254771471
'''