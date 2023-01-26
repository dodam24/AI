import numpy as np          # RNN의 데이터는 3차원 (DNN은 2차원 이상, input_shape는 스칼라의 개수를 벡터 형태로만 넣어줄 것!) (CNN은 데이터 형태 4차원, input_shape는 행 빼고 3차원으로 입력할 것!)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터                                      # 시계열 데이터는 y가 없으므로 만들어줘야 함 (크기에 맞게 자르기)
dataset = np.array([1,2,3,4,5,6,7,8,9,10])      # (10, )    

x = np.array([[1,2,3],                          # (7 X 3)짜리 데이터를 1개씩 잘라서 계산하겠다는 의미
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])
y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape)     # (7, 3), (7,)                                                       # 참고: RNN의 input_shape = (3, )

''' 
x = x.reshape(7, 3, 1)                             
print(x.shape)              # (7, 3, 1) '''


#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(64, input_shape=(3, 1)))
model.add(Dense(64, input_shape=(3, )))
model.add(Dense(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_pred = np.array([8, 9, 10]).reshape(1, 3, 1)      # (3, ) -> 데이터 형태가 맞지 않음 -> reshape 할 것! (N, 3, 1) 형태로
result = model.predict(y_pred)
print('[8, 9, 10]의 결과 : ', result)

# batch 조절, layer 더 깊게, activation 적용 등


''' loss :  0.007506272755563259
[8, 9, 10]의 결과 :  [[11.161505]] '''