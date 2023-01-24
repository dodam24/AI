from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
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
    x, y, shuffle=True,
    random_state=333, 
    test_size=0.2,
    stratify=y
)

# 데이터 전처리
scaler = MinMaxScaler()                     # minmaxscaler 정의
# scaler = StandardScaler()
scaler.fit(x_train)                         # x값의 범위만큼 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)           # x_train fit한 가중치 값 범위에 맞춰서 x_test 데이터 변환
                                            # train 데이터는 fit, transform하고 test 데이터는 transform만!
                                            
# print(y_train)
# print(y_test)

""" #2. 모델 (순차형)
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(4,)))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(3, activation='softmax')) """

#2. 모델 구성(함수형)   # 순차형과 반대로 레이어 구성
input1 = Input(shape=(4,))      # 입력 데이터의 크기(shape)를 Input() 함수의 인자로 입력층 정의
dense1 = Dense(5, activation='linear')(input1)      # 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당
dense2 = Dense(4, activation='sigmoid')(dense1)
dense3 = Dense(3, activation='relu')(dense2)
dense4 = Dense(2, activation='linear')(dense3)
output1 = Dense(3, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)   # 순차형과 달리 model 형태를 마지막에 정의. Model() 함수에 입력과 출력 정의
model.summary()

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
y_predict = np.argmax(y_predict, axis=1)    # array와 비슷한 형태의 input을 넣어주면 가장 큰 원소의 인덱스 반환
print("y_pred(예측값) : ", y_predict)       # 예측값

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : " , y_test)         # 원래 값

acc = accuracy_score(y_test, y_predict)     # 정확도 (y_pred와 y_test의 차이)
print(acc)


""" loss :  10.745396614074707
accuracy :  0.3333333432674408
y_pred(예측값) :  [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
y_test(원래값) :  [0 2 0 1 1 1 0 2 0 2 2 2 2 0 0 0 2 0 2 1 0 2 1 1 0 2 1 1 1 1]
0.3333333333333333 """