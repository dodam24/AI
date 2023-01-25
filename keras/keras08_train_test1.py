import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])   # (10, )
# y = np.array(range(10))                # (10, )

x_train = np.array([1,2,3,4,5,6,7])    # (7, )
x_test = np.array([8,9,10])            # (3, )  -> 한 개의 특성을 가지기 때문에 데이터 셋이 바뀌어도 괜찮음 (행 무시)

y_train = np.array(range(7))
y_test = np.array(range(7,10))

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)    # 훈련용 데이터로 fit 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)      # 평가용 데이터로 evaluate
print('loss : ', loss)
result = model.predict([11])
print('[11]의 결과 : ', result)


""" 결과
loss :  0.5476468205451965
[11]의 결과 :  [[10.1733675  8.142646  10.217098 ]] """
