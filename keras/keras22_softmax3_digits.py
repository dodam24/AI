import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_digits()    # 손글씨로 쓴 숫자를 분류하는 데이터셋
x = datasets.data
y = datasets['target']
# print(x, y)
"""
[[ 0.  0.  5. ...  0.  0.  0.]      # 원-핫 인코딩 하기 전
 [ 0.  0.  0. ... 10.  0.  0.]
 [ 0.  0.  0. ... 16.  9.  0.]
 ...
 [ 0.  0.  1. ...  6.  0.  0.]
 [ 0.  0.  2. ... 12.  0.  0.]
 [ 0.  0. 10. ... 12.  1.  0.]] [0 1 2 ... 8 9 8]
 """
print(x.shape, y.shape)     # (1797, 64) (1797,)
print(np.unique(y, return_counts=True))     # 각 고유 원소의 개수 출력
"""
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
"""

# 시각화
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[5])
plt.show()

# to_categorical 함수: One-hot 인코딩 해주는 함수 (10진 정수 형식을 2진 바이너리 형식(0,1)으로 변경)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
""" 
[[1. 0. 0. ... 0. 0. 0.]        # 원-핫 인코딩 한 결과
 [0. 1. 0. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 1. 0.]
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 1. 0.]]
 """
print(y.shape)                 # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=333,
    stratify=y)     # 분류 모델 다룰 때 사용 (각 층별로 랜덤 데이터 추출)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(64,)))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(100, activation='linear'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)

model.fit(x_train, y_train, epochs=700, batch_size=32, 
          validation_split=0.2, 
          verbose=1, 
          callbacks=[earlyStopping])


#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)               # y_predict (예측값)
y_predict = np.argmax(y_predict, axis=1)

y_test = np.argmax(y_test, axis=1)              # y_test (원래 값)

print('y_pred : ', y_predict)
print('y_test : ', y_test)

from sklearn.metrics import accuracy_score      # accuray (정확도)
acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)


# ValueError: Shapes (None, 10) and (None, 1) are incompatible