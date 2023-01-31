import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
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
    target_size=(200, 200),     # 모든 데이터가 200 x 200으로 증폭 또는 축소됨
    batch_size=10,         
    class_mode='binary',      
    color_mode='grayscale',
    shuffle=True,
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(200, 200),     # 모든 데이터가 200 x 200으로 증폭 또는 축소됨
    batch_size=10,              # 전체 데이터 뽑고 싶으면 batch_size 엄청 크게 줄 것! ex) 데이터 160개, batch_size=10000이면 160개만 나옴
    class_mode='binary',      
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
print(xy_train[0][0].shape)     # (10, 200, 200, 1) -> 맨 앞에 있는 10은 batch_size
print(xy_train[0][1].shape)     # (10,)
print(xy_train[15][0].shape)
print(xy_train[15][1].shape)

print(type(xy_train))           # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))        # <class 'tuple'>
print(type(xy_train[0][0]))     # x는 0의 0번째     # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))     # <class 'numpy.ndarray'>



''' x = (160, 150, 150, 1)    ad: 80장, normal: 80장이므로
y = (160,)
np.unique 결과: 0, 1 각각 80, 80 '''
