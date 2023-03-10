import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# #1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1234)

print(x_train, x_test)  # [15  3 11  8  2 10  9  5  6  7  4 16] [14 12  1 13]   [random]
print(y_train, y_test)  # [15  3 11  8  2 10  9  5  6  7  4 16] [14 12  1 13]   [random]
print(x_train.shape, x_test.shape)  # (12,) (4,)
print(y_train.shape, y_test.shape)  # (12,) (4,)

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(24))
model.add(Dense(11))
model.add(Dense(82))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.25)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print("17의 예측값 : ", result)

""" 
loss :  6.469818458754162e-09
17의 예측값 :  [[17.000011]] 
"""