from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)   # pandas.describe() / pandas.info()
"""  :Summary Statistics:   # x_columns = 4, y_columns = 1
    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ==================== """
print(datasets.feature_names)   # pandas.columns
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets.data
y = datasets['target']

# print(x)
""" 
 [[5.1 3.5 1.4 0.2]
  [4.9 3.  1.4 0.2]
  [4.7 3.2 1.3 0.2]
"""

# print(y)
""" 
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
"""

print(x.shape, y.shape)     # (150, 4), (150,)

from tensorflow.keras.utils import to_categorical   # one hot encoding 통해 y.shape = (150, 3)으로 변경됨
y = to_categorical(y)
print(y)
print(y.shape)
""" 
원-핫 인코딩 한 결과(형태):
[[1. 0. 0]
 [0. 1. 0]
 [0. 0. 1]] """


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=333, 
    test_size=0.2,
    stratify=y          # 데이터 양이 많아질수록 train과 test의 데이터 값이 한 쪽으로 치우쳐서 분류되어 예측 모델의 성능이 하락할 수 있음
                        # 따라서 분류할 때, 한 쪽으로만 데이터가 치우치지 않게 해주기 위해 stratify=y 옵션 사용 (데이터의 종류를 동일한 비율로 추출)
)

# 데이터 전처리 (스케일링)
scaler = MinMaxScaler()                     # minmaxscaler 정의
# scaler = StandardScaler()
scaler.fit(x_train)                         # x값의 범위만큼 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)           # x_train fit한 가중치 값 범위에 맞춰서 x_test 데이터 변환

# print(y_train)
# print(y_test)

#2. 모델
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(4,)))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=1,
          validation_split=0.2,
          verbose=1)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)              # accuracy (정확도)

# print(y_test[:5])
# y_predict = model.predict(x_test[:10])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)       # 예측값

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : " , y_test)         # 원래값

# acc = accuracy_score(y_test, y_predict)
# print(acc)


""" 
loss :  1.1023621559143066
accuracy :  0.3333333432674408
y_pred(예측값) :  [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
y_test(원래값) :  [0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1] """