import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()        # 텐서플로 fashion_mnist 데이터셋 불러와서 변수에 저장

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)  # 뒤에 1 생략 (흑백 데이터)  -> reshape 필요! input_shape = (28, 28, 1)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)


x_train = x_train.reshape(60000, 28*28)     # 2차원으로 변경
x_test = x_test.reshape(10000, 28*28)


print(x_train[0])
print(y_train[0])   # 5

# print(x_train)    # 28 X 28. 6만개    # 흰색(255), 검은색(0)

""" plt.imshow(x_train[0], 'gray')
plt.show """


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input

''' #2. 모델 (DNN 적용)
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784, )))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary() '''

#2. 모델 (함수형)
input1 = Input(shape=(28*28, ))                          # 입력 데이터의 크기(shape)를 Input() 함수의 인자로 입력층 정의
dense1 = Dense(50, activation='linear')(input1)          # 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(10, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)            # 순차형과 달리 model 형태를 마지막에 정의.     Model() 함수에 입력과 출력 정의
model.summary()


#3. 컴파일, 훈련
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32, validation_split=0.25)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])



""" 
이미지 하나는 2차원 구조로 28 * 28 형식으로 이루어져 있음
이미지 밝기에 따라서 각 픽셀은 0 ~ 255 사이의 숫자를 부여 받음
데이터셋이 (x,y,z)로 이루어져 있음. 이미지가 2차원이라서 3차원 데이터 구조를 가지는 것
-> 딥러닝 모델을 돌리기 위해서 데이터셋 구조를 2차원으로 변경 (x,y)
즉, 784개의 속성을 가진 구조로 바꿔야 함 (= 28 * 28)
따라서 (x, 784)로 reshape 해줄 것!

데이터 정규화: 데이터 폭이 클 때, 적절한 값으로 분산의 정도를 바꿈
 - 현재 값인 0 ~ 255를 0 ~ 1 사이의 값으로 변경
 - 최댓값인 255로 x_train 데이터와 x_test 데이터를 나누어 줌
(딥러닝은 데이터셋이 0 ~ 1 사이의 숫자로 되어 있어야 좋은 성능을 발휘하므로)

원-핫 인코딩: 딥러닝의 분류 문제 해결
 - 0 ~ 9까지의 클래스를 0과 1로 이루어진 벡터로 변경
 - to_categorical(클래스, 클래스 개수)
 
"""


""" loss :  0.7213355898857117
acc :  0.720300018787384 """