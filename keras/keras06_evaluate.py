import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))   # 하나의 layer를 표시. Dense(output, input)
model.add(Dense(5))   # hidden layer. (input layer를 뺀 나머지는 명시하지 않음).   # hidden layer 변경 가능
model.add(Dense(4))   # 훈련 수, 노드 개수, layer 깊이 조절 가능 -> 하이퍼파라미터 튜닝
model.add(Dense(2))   
model.add(Dense(1))   # output layer

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=7)   # 훈련

#4. 평가, 예측
loss = model.evaluate(x, y)   # loss값 반환. 3번과 4번의 loss값 비교
print('loss : ', loss)
result = model.predict([6])
print('6의 결과 : ', result)

# loss 값으로 정확성 판단 (predict 값보다 더 중요)

"""
batch size가 1일 때: 5.991158
batch size가 2일 때: 5.976934
batch size가 3일 때: 6.065129
batch size가 4일 때: 5.9726744
batch size가 5일 때: 5.9717484
batch size가 6일 때: 5.969942
batch size가 7일 때: 5.7555237
"""