import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
augument_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augument_size)     # x_train.shape = 60000, 28, 28.    따라서 인덱스 0은 60000이 나옴
print(randidx)
print(len(randidx))     # 40000

x_augument = x_train[randidx].copy()        # 메모리에서 원본을 건드리지 않고 복사본을 넣음.    따라서 x는 40,000개 추출
y_augument = y_train[randidx].copy()        # y도 랜덤으로 40,000개 추출 
print(x_augument.shape, y_augument.shape)        # (40000, 28, 28) (40000,)

x_augument = x_augument.reshape(40000, 28, 28, 1)   # 이미지 데이터로 reshape



train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    # zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

x_augumented = train_datagen.flow(        # directory가 아닌 데이터셋에서 불러옴
    x_augument,
    y_augument,
    # np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1),   # x     # -1은 전체 데이터를 의미
    # np.zeros(augument_size),                                                    # y
    batch_size=augument_size,
    shuffle=True, 
)

# 변환된 40000개의 데이터
print(x_augumented[0][0].shape)   # (40000, 28, 28, 1)
print(x_augumented[0][1].shape)   # (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)

# x_augumented와 x_train 합치기 (concatenate)

x_train = np.concatenate((x_train, x_augumented[0][0]))
y_train = np.concatenate((y_train, x_augumented[0][1]))


print(x_train.shape, y_train.shape)     # 


# 60,000개의 데이터에서 40,000개의 데이터 추출해서 concatenate하면 100,000개
