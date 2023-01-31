# https://kaggle.com/competitions/dogs-vs-cats/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,     
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,     # 원래 이미지의 20% 확대
    # shear_range=0.7,
    # fill_mode='nearest'
    )
# test_datagen = ImageDataGenerator(       # test 데이터는 rescale만 수행 (증폭되지 않은 원 데이터 사용)
#     rescale=1./255
# )

xy_train = train_datagen.flow_from_directory(
    'C:/_data/dogs-vs-cats/train/',
    target_size=(200, 200),   
    batch_size=10000,         
    class_mode='binary',
    # class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    # Found 25000 images belonging to 2 classes.
)

# xy_test = test_datagen.flow_from_directory(
#     'C:/_data/dogs-vs-cats/test/',
#     target_size=(200, 200), 
#     batch_size=10,             
#     class_mode='binary',        
#     # class_mode='categorical',      
#     color_mode='rgb',
#     shuffle=True,
#     # Found 12500 images belonging to 1 classes.
# )

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000022468C22B20>

# print('x : ', xy_train[0][0])
# print('y : ', xy_train[0][1])

print(xy_train[0][0].shape)     # (10000, 200, 200, 3)
print(xy_train[0][1].shape)     # (10000,)

np.save('C:/_data/dogs-vs-cats/train/kaggle_x_train.npy', arr=xy_train[0][0])      
np.save('C:/_data/dogs-vs-cats/train/kaggle_y_train.npy', arr=xy_train[0][1])      
    
# np.save('C:/_data/dogs-vs-cats/test/kaggle_x_test.npy', arr=xy_test[0][0])
# np.save('C:/_data/dogs-vs-cats/test/kaggle_y_test.npy', arr=xy_test[0][1])        