import tensorflow as tf
print(tf.__version__)   # 2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

if(gpus):
    print("gpu 실행함")
else:
    print("gpu 실행안함")