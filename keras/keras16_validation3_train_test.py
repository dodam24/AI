import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
""" 
# [실습]
# train_test_split으로 자르기
# 10:3:3으로 나누기
x_train = np.array(range(1, 11))
y_train = np.array(range(1, 11))
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])

# 검증 데이터 추가 (validation)
x_validation = np.array([14,15,16])
y_validation = np.array([14,15,16]) 
"""

x = np.array([range(1, 17)])
y = np.array([range(1, 17)])


# train_test_split으로 데이터 분리 (train, test, validation)
x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.6, shuffle=True, random_state=123)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,
    test_size=0.4, shuffle=True, random_state=123)

print(x_train, x_test, y_train, y_test, x_validation, y_validation)


#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(9))
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
ValueError: With n_samples=1, test_size=None and train_size=0.6, the resulting train set will be empty. Adj
ust any of the aforementioned parameters. 
"""