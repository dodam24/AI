import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1, 11))
timesteps = 5

def split_x(dataset, timesteps):    # split_x라는 이름으로 함수를 정의 
    aaa = []    # aaa 라는 빈 리스트 생성
    for i in range(len(dataset) - timesteps + 1):   
        subset = dataset[i : (i + timesteps)]       
        aaa.append(subset)                          
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)
''' 
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]] '''
print(bbb.shape)    # (6, 5)

x = bbb[:, :-1]
y = bbb[:, -1]  # 가장 오른쪽 끝
print(x, y)
''' 
[[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]] [ 5  6  7  8  9 10] '''

print(x.shape, y.shape)     # (6, 4) (6,)
                            # x는 2차원이므로 3차원으로 변경 (reshape 이용)

x = x.reshape(6, 4, 1)
print(x.shape)
                            
# 실습
# LSTM 모델 구성

#2. 모델 구성
model = Sequential()
model.add(LSTM(64, input_shape=(4, 1),
               return_sequences=True))
model.add(LSTM(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
x_predict = np.array([7, 8, 9, 10]).reshape(1, 4, 1)     # 11 예측하기
result = model.predict(x_predict)
print('[7, 8, 9, 10]의 결과 : ', result)