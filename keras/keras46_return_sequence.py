import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],                          # (N, 3, 1) -> reshape 할 것!
              [4,5,6],[5,6,7],[6,7,8],
              [7,8,9],[8,9,10],[9,10,11],
              [10,11,12],[20,30,40],
              [30,40,50],[40,50,60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])    # (7 )

print(x.shape, y.shape)     # (13, 3) (13,)
x = x.reshape(13, 3, 1)
print(x.shape)

#2. 모델 구성
model = Sequential()    # input_shape = (N, 3, 1)
model.add(LSTM(64, input_shape=(3, 1),     # (N, 64) -> 여기서 2차원을 전달하는데 아래에 있는 LSTM은 3차원을 받아들임
               return_sequences=True))     # LSTM 2개 이상 사용할 때, return_sequences=True 옵션 사용 
model.add(LSTM(32))     # LSTM은 3차원을 받아들여서 2차원을 전달함
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=2)

#4. 평가, 예측 
loss = model.evaluate(x, y)
print('loss : ', loss)
x_predict = np.array([50, 60, 70]).reshape(1, 3, 1)        # 80 예측하기
result = model.predict(x_predict)
print('[50, 60, 70]의 결과 : ', result)