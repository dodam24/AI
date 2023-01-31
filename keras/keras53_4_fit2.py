# matplotlib으로 그림 그리기

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터
train_datagen = ImageDataGenerator(         # 데이터가 크면 연산이 오래 걸리므로 데이터의 크기를 줄임
    rescale=1./255, 
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,     # 원래 이미지의 20% 확대
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(       # test 데이터는 rescale만 수행 (증폭되지 않은 원 데이터 사용)
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(       # 이름이 xy_인 이유: x, y 묶어서 데이터 셋 구성
    './_data/brain/train/',
    target_size=(100, 100),     # 모든 데이터가 200 x 200으로 증폭 또는 축소됨
    batch_size=1000,            # batch_size 크게 줘서 전체 데이터 개수 추출 가능 (데이터 개수만큼만 돌기 때문에)
    class_mode='binary',      
    color_mode='grayscale',
    shuffle=True,
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(100, 100),     # 모든 데이터가 200 x 200으로 증폭 또는 축소됨
    batch_size=10,              # 전체 데이터 뽑고 싶으면 batch_size 엄청 크게 줄 것! ex) 데이터 160개, batch_size=10000이면 160개만 나옴
    class_mode='binary',      
    color_mode='grayscale',
    shuffle=True,
    # Found 160 images belonging to 2 classes.
)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(100, 100, 1)))     # (99, 99, 64)
model.add(Conv2D(64, (3,3), activation='relu'))             # (97, 97, 64)
model.add(Conv2D(32, (3,3), activation='relu'))             # (95, 95, 32)
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))                   # y가 0, 1이므로 sigmoid 적용 (또는 softmax 사용해서 one-hot encoding 해줄 것!)
# model.add(Dense(2, activation='softmax'))                 # sigmoid 대신 softmax 함수 적용 (one-hot encoding 해줄 것! 이 때, compile의 loss='sparse_categorical_crossentropy' 적용)


#3. 컴파일, 훈련 (validation_split 이용해서 코드 수정)
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])      # sigmoide 대신 softmax 함수 적용했을 때 
''' hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=300,    # 160개 데이터를 batch_size=10으로 했으니까 steps_per_epochs의 값은 16임
                    validation_data = xy_test,
                    validation_steps=4) '''
hist = model.fit(# xy_train[0][0], xy_train[0][1],
                 xy_train,
                 batch_size=16, 
                 epochs=300, 
                 validation_split=0.3)
                    

accuracy = hist.history['acc']              # 훈련에 대한 accuracy, val_acc, loss, val_loss (evaluate 아님)
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])


''' #4. 평가, 예측
loss = model.evaluate(xy_test)
print('loss : ', loss)
'''

# 이미지 데이터는 gpu로 돌려야 빠름