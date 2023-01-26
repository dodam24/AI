import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

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

print(x.shape, y.shape)     # (7, 3) (7,)                                                      
x = x.reshape(7, 3, 1)                              
                                                    
print(x.shape)              # (7, 3, 1)             

#2. 모델 구성               # RNN과 LSTM 동일 (둘 중에 성능 좋은 것으로 선택)
model = Sequential() 
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))                    # (N, 3, 1) -> (batch, timesteps, feature)
model.add(LSTM(units=10, input_shape=(3, 1)))
# model.add(LSTM(units=10, input_length=3, input_dim=1))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.summary()

# simple RNN
# 10 * (10 + 1 + 1) = 120
''' _________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn (SimpleRNN)      (None, 10)                120

 dense (Dense)               (None, 5)                 55

 dense_1 (Dense)             (None, 1)                 6

================================================================= '''

# LSTM
# simpleRNN * 4 = 120 * 4 = 480
''' _________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 10)                480

 dense (Dense)               (None, 5)                 55

 dense_1 (Dense)             (None, 1)                 6

================================================================= '''
