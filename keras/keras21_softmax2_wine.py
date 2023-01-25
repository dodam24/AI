import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_wine()      # 와인의 등급 분류
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (178, 13) (178,)
print(y)
""" 
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
 """
print(np.unique(y))                         # array에서 unique한 원소만 추출: [0 1 2]
print(np.unique(y, return_counts=True))     # 각 고유 원소의 개수를 출력
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64)) : 0은 59개, 1은 71개, 2는 48개라는 의미

# to_categorical 함수는 One-hot 인코딩 해주는 함수
# 원-핫 인코딩: 10진 정수 형식을 특수한 2진 바이너리 형식으로 변경 (0, 1)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
""" 
[[1. 0. 0.]     # y를 출력하면 이런 식의 형태임
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]] 
 """
print(y.shape)     # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=333,
    test_size=0.2,
    stratify=y      # 계층적 데이터 추출 옵션 (여러 층으로 분할 후, 각 층별로 랜덤 데이터 추출. 원래 데이터의 분포와 유사하게 데이터 추출)
)

#2. 모델
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(13,)))
model.add(Dense(42, activation='sigmoid'))
model.add(Dense(35, activation='relu'))
model.add(Dense(21, activation='linear'))
model.add(Dense(3, activation='softmax'))   # 다중 분류: activation='softmax'

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중 분류: loss='categorical_crossentropy'
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2,
          verbose=1,
          callbacks=[earlystopping])

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

from sklearn.metrics import accuracy_score     # accuracy_score(정확도)
import numpy as np

y_predict = model.predict(x_test)              # 예측값
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : " , y_test)            # 원래값

acc = accuracy_score(y_test, y_predict)        # 정답률 (=정확도): 실제 데이터 중 맞게 예측한 데이터의 비율
print('acc : ', acc)


""" 
loss :  0.5188944935798645
accuracy :  0.6944444179534912
y_pred(예측값) :  [1 0 1 0 1 2 2 0 0 1 1 1 1 0 2 2 1 0 1 0 1 1 0 0 0 0 0 2 1 1 2 2 1 0 2 1]
y_test(원래값) :  [1 0 1 0 1 1 1 0 0 1 1 1 2 0 2 1 2 1 1 0 1 2 0 0 0 0 0 2 2 2 1 2 2 0 2 1]
acc :  0.6944444444444444 
"""