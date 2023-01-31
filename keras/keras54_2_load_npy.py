import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255)
''' horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
'''

test_datagen = ImageDataGenerator(       # test 데이터는 rescale만 수행 (증폭되지 않은 원 데이터 사용)
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(       # 이름이 xy_인 이유: x, y 묶어서 데이터 셋 구성
    './_data/brain/train/',
    target_size=(200, 200),     # 모든 데이터가 200 x 200으로 증폭 또는 축소됨
    batch_size=10000,         
    class_mode='binary',        # 원-핫 할 필요 없으므로 binary 사용
    # class_mode='categorical',
    color_mode='grayscale',
    shuffle=True,
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(200, 200),     # 모든 데이터가 200 x 200으로 증폭 또는 축소됨
    batch_size=10,              # 전체 데이터 뽑고 싶으면 batch_size 엄청 크게 줄 것! ex) 데이터 160개, batch_size=10000이면 160개만 나옴
    class_mode='binary',        # 원-핫 할 필요 없으므로 binary 사용 
    # class_mode='categorical',      
    color_mode='grayscale',
    shuffle=True,
    # Found 120 images belonging to 2 classes.
)

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000022468C22B20>

''' from sklearn.datasets import load_iris
datasets = load_iris()
print(datasets) '''

# print(xy_train[0])
# print(xy_train[0][0])
# print(xy_train[0][1])
# print(xy_train[0][0].shape)     # (10, 200, 200, 1)
# print(xy_train[0][1].shape)     # (10, 2)

# np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])      
# np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])      

# np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])        
# np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])

x_train = np.load('./_data/brain/brain_x_train.npy')
y_train = np.load('./_data/brain/brain_y_train.npy')
x_test = np.load('./_data/brain/brain_x_test.npy')
y_test = np.load('./_data/brain/brain_y_test.npy')

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


''' print(type(xy_train))           # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))        # <class 'tuple'>
print(type(xy_train[0][0]))     # x는 0의 0번째     # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))     # <class 'numpy.ndarray'> '''


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(200, 200, 1)))     # (99, 99, 64)
model.add(Conv2D(64, (3,3), activation='relu'))             # (97, 97, 64)
model.add(Conv2D(32, (3,3), activation='relu'))             # (95, 95, 32)
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])      # sigmoide 대신 softmax 함수 적용했을 때                  
hist = model.fit(xy_train[0][0], xy_train[0][1], 
                 batch_size=16, 
                 epochs=300, 
                 validation_data = (xy_test[0][0], xy_test[0][1]))          # 전체 데이터를 batch 1개로 잡아서 x_train과 y_train을 다음과 같은 형태로 표현 가능
''' hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=300,    # 160개 데이터를 batch_size=10으로 했으니까 steps_per_epochs는 16
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



''' loss :  1.3590212120107026e-06
val_loss :  7.047504368529189e-06
accuracy :  1.0
val_acc :  1.0 '''