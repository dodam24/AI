import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()        # 텐서플로 mnist 데이터셋 불러와서 변수에 저장

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)  # 뒤에 1 생략 (흑백 데이터)  
                                        # reshape 해야 함. input_shape = (28, 28, 1)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)


print(x_train[0])
print(y_train[0])   # 5

# print(x_train) # 28 X 28. 6만개    # 흰색(255), 검은색(0)

plt.imshow(x_train[0], 'gray')
plt.show