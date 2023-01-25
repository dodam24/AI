import numpy as np

import tensorflow as tf
print(tf.__version__)


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#1. (정제된) 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

#2. 모델 구성
from tensorflow.keras.models import Sequential
# tensorflow에 있는 keras 문법을 사용해 sequential 모델을 가져옴
# keras 모델: sequential 모델, 함수 API에 사용되는 Model
# sequential은 레이어를 선형으로 연결하여 구성하는, 케라스에서 가장 단순한 신경망 모델

from tensorflow.keras.layers import Dense
# tensorflow에 있는 keras 문법을 사용해 Dense 계층(레이어)을 가져옴
# keras 계층: Dense, Activation, Dropout, Lambda 등

model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')     # compile: 사람의 언어를 컴퓨터가 이해할 수 있는 언어로 바꿔주는 과정
model.fit(x, y, epochs=1000)                    # fit: 모델을 학습 시키는 것

#4. 평가, 예측
result = model.predict([13])
print('결과 : ', result)        # 결과 :  [[12.998622]]