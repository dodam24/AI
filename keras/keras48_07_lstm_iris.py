from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
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

print(x_train.shape, x_test.shape)      # (120, 4) (30, 4)

x_train = x_train.reshape(120, 4, 1)       
x_test = x_test.reshape(30, 4, 1)
print(x_train.shape, x_test.shape)


#2. 모델 구성 (순차형)
model = Sequential()
model.add(LSTM(64, input_shape=(4, 1)))
model.add(Dense(3, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=1,
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