import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],        # (N, 3, 1) -> reshape 할 것!
              [4,5,6],[5,6,7],[6,7,8],
              [7,8,9],[8,9,10],[9,10,11],
              [10,11,12],[20,30,40],
              [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])    # (7 )

print(x.shape, y.shape)     # (13, 3) (13,)
x = x.reshape(13, 3, 1)
print(x.shape)

#2. 모델 구성
model = Sequential()
model.add(LSTM(units=10, input_shape=(3, 1)))
model.add(Dense(5, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(15))
model.add(Dense(10, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=2)

#4. 평가, 예측 
loss = model.evaluate(x, y)
print('loss : ', loss)
x_predict = np.array([50, 60, 70]).reshape(1, 3, 1)        # 80 예측하기
result = model.predict(x_predict)
print('[50, 60, 70]의 결과 : ', result)