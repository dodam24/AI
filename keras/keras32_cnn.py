from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten      


# 모델 구성
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), 
                 input_shape=(10,10,1)))              
model.add(Conv2D(5, kernel_size=(2,2)))             
model.add(Conv2D(7, (2,2)))
model.add(Conv2D(6, 2))     # 2는 커널 사이즈=(2,2)의 의미
model.add(Flatten())                                
model.add(Dense(units=10))           
model.add(Dense(1, activation='relu'))

model.summary()