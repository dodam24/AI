import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print(y)
print(np.unique(y)) # [0 1 2]
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# 원-핫 인코딩
from tensorflow.keras.utils import to_categorical   # (178, 3)으로 변경
y = to_categorical(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=333,
    test_size=0.2,
    stratify=y
)

# 데이터 전처리 (스케일링)
scaler = MinMaxScaler()                     # minmaxscaler 정의
# scaler = StandardScaler()
scaler.fit(x_train)                         # x값의 범위만큼 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)           # x_train fit한 가중치 값 범위에 맞춰서 x_test 데이터 변환
                                            # train 데이터는 fit, transform하고 test 데이터는 transform만!

""" #2. 모델 (순차형)
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(13,)))
model.add(Dense(42, activation='sigmoid'))
model.add(Dense(35, activation='relu'))
model.add(Dense(21, activation='linear'))
model.add(Dense(3, activation='softmax'))   # 다중 분류: activation='softmax' """

#2. 모델 구성(함수형)   # 순차형과 반대로 레이어 구성
input1 = Input(shape=(13,))     # 입력 데이터의 크기(shape)를 Input() 함수의 인자로 입력층 정의
dense1 = Dense(50, activation='linear')(input1)     # 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당
dense2 = Dense(42, activation='sigmoid')(dense1)
dense3 = Dense(35, activation='relu')(dense2)
dense4 = Dense(21, activation='linear')(dense3)
output1 = Dense(3, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)   # 순차형과 달리 model 형태를 마지막에 정의.     Model() 함수에 입력과 출력 정의
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중 분류: loss='categorical_crossentropy'
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2,
          verbose=1,
          callbacks=[earlyStopping])

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

from sklearn.metrics import accuracy_score
import numpy as np

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : " , y_test)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)


""" Epoch 00064: early stopping
loss :  0.08350849151611328
accuracy :  0.9722222089767456
y_pred(예측값) :  [1 0 1 0 0 1 1 0 0 1 1 1 2 0 2 1 2 1 1 0 1 2 0 0 0 0 0 2 2 2 1 2 2 0 2 1]
y_test(원래값) :  [1 0 1 0 1 1 1 0 0 1 1 1 2 0 2 1 2 1 1 0 1 2 0 0 0 0 0 2 2 2 1 2 2 0 2 1]
acc :  0.9722222222222222 """