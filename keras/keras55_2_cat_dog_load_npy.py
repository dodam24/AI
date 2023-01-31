import numpy as np

x_train = np.load('C:/_data/dogs-vs-cats/train/kaggle_x_train.npy')
y_train = np.load('C:/_data/dogs-vs-cats/train/kaggle_y_train.npy')
# x_test = np.load('C:/_data/dogs-vs-cats/test/kaggle_x_test.npy')
# y_test = np.load('C:/_data/dogs-vs-cats/test/kaggle_y_test.npy')

print(x_train.shape)      # (10, 200, 200, 3) (10, 200, 200, 3)
print(y_train.shape)      # (10,) (10,)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(16, (2,2), input_shape=(200, 200, 3)))    
model.add(Conv2D(2, (10,10), padding='same', activation='relu'))             
model.add(Conv2D(16, (3,3), activation='relu'))            
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])                  
hist = model.fit(x_train, y_train, 
                 batch_size=32, 
                 epochs=3,
                 validation_split=0.1)
''' hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=300,
                    validation_data = xy_test,
                    validation_steps=4) '''
                    

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])