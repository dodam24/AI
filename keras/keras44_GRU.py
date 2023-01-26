import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

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
# model.add(LSTM(units=10, input_shape=(3, 1)))
model.add(GRU(units=10, input_shape=(3, 1)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.summary()

# GRU
# 3 * units * (feature + bias + units + ?) = param
''' _________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 gru (GRU)                   (None, 10)                390

 dense (Dense)               (None, 5)                 55

 dense_1 (Dense)             (None, 1)                 6

================================================================= '''
