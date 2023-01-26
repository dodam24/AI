import numpy as np                               # CNN: 4차원    DNN: 2차원    RNN: 3차원
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터                                      
dataset = np.array([1,2,3,4,5,6,7,8,9,10])      # (10, )    

x = np.array([[1,2,3],                          
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])
y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape)     # (7, 3), (7,)                                                      
x = x.reshape(7, 3, 1)                              
                                                    
print(x.shape)              # (7, 3, 1)             

#2. 모델 구성
model = Sequential()
model.add(SimpleRNN(units = 64, input_shape=(3, 1)))
                                # (N, 3, 1) -> (batch, timesteps, feature)      # timesteps만큼 잘라서 feature만큼 일 시키기
model.add(Dense(16, activation='relu'))
model.add(Dense(20))
model.add(Dense(15, activation='relu'))
model.add(Dense(10))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# 64  * (64 + 1 + 1) = 4224
# units = (feature + bias + units) = param