import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense


#1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))

# 데이터 쪼개기 (4:1)
timesteps = 5

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):   
        subset = dataset[i : (i + timesteps)]       
        aaa.append(subset)                          
    return np.array(aaa)

bbb = split_x(a, timesteps)     # 데이터를 5개씩 자름
# print(bbb)
# print(bbb.shape)      # (96, 5)

x = bbb[:, :-1]         # x는 앞에서 4개
y = bbb[:, -1]          # y는 맨 뒤 1개
print(x, y)

print(x.shape, y.shape)     # (96, 4) (96,)
x = x.reshape(96, 4, 1)     # RNN에 넣어줘야 하므로 3차원 형태로 변경


# 결과(x_predict) 쪼개기 
timesteps = 4

ccc = split_x(x_predict, timesteps)
print(ccc.shape)        # (7, 4)

x_predict = ccc.reshape(7, 4, 1)
        
        
#2. 모델 구성
model = Sequential()
model.add(Conv1D(100, 2, input_shape=(4, 1), padding='same'))   # 2는 커널 사이즈
model.add(Conv1D(64, 2, input_shape=(4, 1), padding='same'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=5)


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x_predict)
print('[100, 101, 102, 103, 104, 105, 106]에 대한 예측값 : ')
print(result)



""" 
loss :  0.004346523433923721
[100, 101, 102, 103, 104, 105, 106]에 대한 예측값 : 
[[100.12431 ]
 [101.12666 ]
 [102.128975]
 [103.13131 ]
 [104.13363 ]
 [105.13597 ]
 [106.13828 ]] """