import tensorflow as tf     # 텐서플로 가져오기 (tf로 명칭)
print(tf.__version__)
import numpy as np

#1. 데이터 준비
x = np.array([1,2,3])   # numpy 형식의 행렬
y = np.array([1,2,3])   

#2. 인공지능 모델 구성 (케라스 문법을 통한 텐서플로)
from tensorflow.keras.models import Sequential  # 딥러닝 순차적 모델
from tensorflow.keras.layers import Dense   # y = wx+b 구성을 위한 기초

model = Sequential()    # 순차적 모델 구성. layer에 순차적으로 연산
model.add(Dense(1, input_dim=1))    # dim: dimension(x, y) 한 덩어리
                                    # Dense(y, x) 형태. y: output, x: input
                                    # 1 = output_dim(y, 출력). input_dim(x 행렬) 한 덩어리 1로 추가

# 최적의 weight와 bias를 찾기 위한 "정제된" 데이터가 중요

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')     # loss 값을 낮추기 위해 mae 사용
                                                # loss를 최적화하기 위해 adam 사용
model.fit(x, y, epochs=2000)    # fit은 데이터 훈련
                                # epochs: 훈련을 몇 번 시킬지

#4. 평가, 예측
result = model.predict([4]) # 결과 예측 (4에 대한 예측 값이 몇이 나올지)
print('결과 : ', result)
