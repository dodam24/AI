import tensorflow as tf #텐서플로 가져오기 (tf로 명칭)
print(tf.__version__)
import numpy as np

#1. 데이터 준비
x = np.array([1,2,3]) #x = 1,2,3
y = np.array([1,2,3]) #y = 1,2,3

#2. 인공지능 모델 구성
from tensorflow.keras.models import Sequential #순차적 모델
from tensorflow.keras.layers import Dense #구성(?)

model = Sequential() #모델
model.add(Dense(1, input_dim=1)) #데이터 //dim = dimension (x, y) 한 덩어리 //Dense(y(output),x(input))
 
#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') #컴파일
 # mae:mean average error //loss값을 낮추기 위해 mae 사용, loss를 최적화하기 위해 adam 사용
model.fit(x, y, epochs=2000) #x, y 데이터 훈련, 에포:훈련을 몇 번 시킬지(2000번)

#4. 평가, 예측
result = model.predict([4]) #4에 대한 값이 몇이 나올지(결과) 예측
print('결과 : ', result)