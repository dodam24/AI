import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
# array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[5])
plt.show()

# 원-핫 인코딩
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=333)

# 데이터 전처리
scaler = MinMaxScaler()                     # minmaxscaler 정의
# scaler = StandardScaler()
scaler.fit(x_train)                         # x값의 범위만큼 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)           # x_train fit한 가중치 값 범위에 맞춰서 x_test 데이터 변환
                                            # train 데이터는 fit, transform하고 test 데이터는 transform만!

""" #2. 모델 구성 (순차형)
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(64,)))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(9, activation='linear'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(10, activation='softmax')) """

#2. 모델 구성 (함수형)  # 순차형과 반대로 레이어 구성
input1 = Input(shape=(64,))     # 입력 데이터의 크기(shape)를 Input() 함수의 인자로 입력층 정의
dense1 = Dense(10, activation='linear')(input1)     # 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당
dense2 = Dense(5, activation='sigmoid')(dense1)
dense3 = Dense(9, activation='relu')(dense2)
dense4 = Dense(7, activation='linear')(dense3)
dense5 = Dense(3, activation='relu')(dense4)
output1 = Dense(10, activation='linear')(dense5)
model = Model(inputs=input1, outputs=output1)   # 순차형과 달리 model 형태를 마지막에 정의
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)
import time
start = time.time()
model.fit(x_train, y_train, epochs=700, batch_size=32, validation_split=0.2, verbose=1, callbacks=[earlyStopping])
end = time.time()

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
print('y_pred : ', y_predict)
print('y_test : ', y_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)
print('time : ', end - start)