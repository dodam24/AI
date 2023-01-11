from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)   # pandas.describe() / pandas.info()
print(datasets.feature_names)   # pandas.columns

x = datasets.data
y = datasets['target']
print(x)
print(y)
print(x.shape, y.shape) # (150, 4), (150,)

from tensorflow.keras.utils import to_categorical   # one hot encoding 통해 y.shape = (150, 3)으로 변경됨
y = to_categorical(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, # False의 문제점 : 
    random_state=333, 
    test_size=0.2,
    stratify=y
)

scaler = MinMaxScaler() # minmaxscaler 정의
#  scaler = StandardScaler()
scaler.fit(x_train) # x값의 범위만큼의 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)# 시작 (transform해야 바뀐다.)

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
print('accuracy : ', accuracy)

# print(y_test[:5])
# y_predict = model.predict(x_test[:10])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)
y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : " , y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)