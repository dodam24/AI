import numpy as np

#1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))    # 예상: y = 100 ~ 107 (split 해야 함)


timesteps1 = 5       # x는 4개, y는 1개
timesteps2 = 4

def split_x(dataset, timesteps1):    # split_x라는 이름으로 함수를 정의 
    aaa = []    # aaa 라는 빈 리스트 생성
    for i in range(len(dataset) - timesteps1 + 1):   
        subset = dataset[i : (i + timesteps1)]       
        aaa.append(subset)                          
    return np.array(aaa)

bbb = split_x(a, timesteps1)
print(bbb)
print(bbb.shape)    # (96, 5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)

print(x.shape, y.shape)     # (96, 4) (96,)
x = x.reshape(96, 4, 1)


def split_y(dataset, timesteps2):
    ccc = []   
    for i in range(len(dataset) - timesteps2 + 1):   
        subset = dataset[i : (i + timesteps2)]       
        ccc.append(subset)                          
    return np.array(ccc)


x_predict = split_y(x_predict, timesteps2)
print(x_predict)
print(x_predict.shape)                  # (7, 4)

x_predict = x_predict.reshape(7, 4, 1)
print(x_predict.shape)                  # (7, 4, 1)
  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123)

''' x_train = x_train.reshape(72, 4, 1)
x_test = x_test.reshape(24, 4, 1)
x_predict = x_predict.reshape(7, 4, 1) '''

x_train = x_train.reshape(72, 2, 2)
x_test = x_test.reshape(24, 2, 2)
x_predict = x_predict.reshape(7, 2, 2)



#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 

model = Sequential()
model.add(LSTM(64, input_shape=(2, 2),
               return_sequences=True))
model.add(LSTM(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# x_predict = np.array((range(96, 106))).reshape()     # 100 ~ 107 예측

result = model.predict(x_predict)
print('결과 : ', result)