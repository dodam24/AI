import numpy as np

# 모델 save
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
     rescale=1./255,
    # horizontal_flip=True,     
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest'
)

xy_train = train_datagen.flow_from_directory(
    'C:/_data/rps/rps/',
    target_size=(200, 200),   
    batch_size=10000,         
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    # Found 2520 images belonging to 1 classes.
)

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001DEDC347F70>

# print('x : ', xy_train[0][0])
# print('y : ', xy_train[0][1])
''' 
y :  [[0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 ...
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]] '''

print(xy_train[0][0].shape)    # (2520, 200, 200, 3)
print(xy_train[0][1].shape)    # (2520, 3)

np.save('C:/_data/rps/rps/rps_x_train.npy', arr=xy_train[0][0])      
np.save('C:/_data/rps/rps/rps_y_train.npy', arr=xy_train[0][1]) 


# load 모델
x_train = np.load('C:/_data/rps/rps/rps_x_train.npy')
y_train = np.load('C:/_data/rps/rps/rps_y_train.npy')

print(x_train.shape)
print(y_train.shape)




''' y_predict = model.predict(xy_train)
y_predict = np.argmax(y_predict, axis=1) '''