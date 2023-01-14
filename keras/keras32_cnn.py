from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten      # Conv2D: 2차원 이미지 데이터를 다루기 위해 사용
                                                                # Flatten: 2차원 데이터를 1차원으로 펴주는 것
model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(2,2), 
                 input_shape=(5,5,1)))  # 가로, 세로, 컬러 (5 X 5의 '흑백' 이미지 1장)    # 1: 흑백, 3: 컬러(RGB)
                                        # if (60000, 5, 5, 1)의 형태일 경우, (5, 5, 1) 짜리가 60,000장이라는 뜻
                                        # (batch_size, rows, columns, channels)
model.add(Conv2D(5,(2,2)))      # filter 수는 여러 번 수행해보며 적절한 값을 찾아야 함
# model.add(Conv2D(filters=5, kernel_size=(2,2))
model.add(Flatten())    # (N, 3, 3, 5)를 1차원으로 펼쳐서 (N, 45)의 형태로 변환
model.add(Dense(units=10))  # (N, 10)   # input은 (batch_size, input_dim)  # input_dim = column 개수
model.add(Dense(1, activation='relu'))

model.summary()

""" _________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50

 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205

 flatten (Flatten)           (None, 45)                0

 dense (Dense)               (None, 10)                460

 dense_1 (Dense)             (None, 1)                 11

=================================================================
Total params: 726
Trainable params: 726
Non-trainable params: 0
_________________________________________________________________ """