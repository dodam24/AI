import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255)
''' horizontal_flip=True,       # 변환본 가지고 있는 것보다 원본 데이터를 가지고 있는 것이 더 좋으므로 전체 주석 처리
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,     # 원래 이미지의 20% 확대
    shear_range=0.7,
    fill_mode='nearest
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
print(xy_train[0][1])
print(xy_train[0][0].shape)     # (10, 200, 200, 1)
print(xy_train[0][1].shape)     # (10, 2)

np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])      # 이미지 파일들을 numpy 형태로 저장.   x_train 데이터가 들어감
np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])      # y_train 데이터가 들어감
# np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0])       # 전체 데이터가 들어감 (x_train, y_train 나누지 말고 한 번에 save도 가능)

np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])        # 이미지 파일들을 numpy 형태로 저장.   x_test 데이터가 들어감
np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])        # y_test 데이터가 들어감
# np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0])         # # 전체 데이터가 들어감 (x_test, y_test 나누지 말고 한 번에 save도 가능)

''' print(type(xy_train))           # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))        # <class 'tuple'>
print(type(xy_train[0][0]))     # x는 0의 0번째     # <class 'numpy.ndarray'>
print(type(xy_train[0][1]))     # <class 'numpy.ndarray'> '''