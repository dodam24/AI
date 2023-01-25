import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# #1. 데이터
x = np.array([range(1, 17)])    # [[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]]   # (1, 16)
y = np.array([range(1, 17)])    # [[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]]   # (1, 16)


"""
x_train = np.array(range(1, 11))    # [ 1  2  3  4  5  6  7  8  9 10]   # (10,)
y_train = np.array(range(1, 11))    # [ 1  2  3  4  5  6  7  8  9 10]   # (10,)

x_test = np.array([11,12,13])               # (3,)
y_test = np.array([11,12,13])               # (3,)

# 검증 데이터 추가 (머신이 훈련한 것을 검증하는 것)
x_validation = np.array([14,15,16])         # (3,)
y_validation = np.array([14,15,16])         # (3,)

"""
 
# 상기 데이터와 동일한 데이터 (x_train, y_train, x_test, y_test, x_validation, y_validation)
# [실습] 슬라이싱으로 데이터 분리
x_train = x[:10]        # 인덱스 0 ~ 9번까지: [1  2  3  4  5  6  7  8  9 10]
x_test = x[10:13]       # 인덱스 10 ~ 12번:   [10 11 12]
y_train = y[:10]        # 인덱스 0 ~ 9번까지: [1  2  3  4  5  6  7  8  9 10]
y_test = y[10:13]       # 인덱스 10 ~ 12번:   [10 11 12]
x_validation = x[13:]   # 인덱스 13 ~ 끝까지:  [14 15 16]
y_validation = x[13:]   # 인덱스 13 ~ 끝까지:  [14 15 16]


#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=3,
          validation_data=(x_validation, y_validation))     # 검증 데이터 추가

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)


""" 
ValueError: Exception encountered when calling layer "sequential" (type Sequential).

Input 0 of layer "dense" is incompatible with the layer: expected axis -1of input shape to have value 1
, but received input with shape (None, 16) 
"""