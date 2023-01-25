import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))

""" 
# 1. keras: to_categorical
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)      # (58102, 8)
print(type(y))      # <class 'numpy.ndarray'>
print(y[:10])       # 데이터 10개만 찍어보기
print(np.unique(y[:,0], return_counts=True))
    # (array([0.], dtype=float32), array([581012], dtype=int64))
print(np.unique(y[:,1], return_counts=True))
    # (array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))
print("===================================")
y = np.delete(y, 0, axis=1)     # axis=1(열), axis=0(행) 삭제
print(y.shape)      # (581012, 7)
print(y[:10])       # 데이터 10개만 찍어보기
print(np.unique(y[:,0], return_counts=True))    # 각 고유 값과 개수 출력
    # (array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64)) """


""" 
# 2. pandas: get_dummies
import pandas as pd
y = pd.get_dummies(y)
print(y[:10])
print(type(y))      # <class 'pandas.core.frame.DataFrame'>
y = y.to_numpy()    # pandas형 데이터를 numpy 형태로 변경
print(type(y))      # <class 'numpy.ndarray'>
print(y.shape)      # (581012, 7)

# 판다스는 헤더와 인덱스가 존재함
# 인덱스와 헤더(컬럼명)는 연산에 사용되지 않음 """


""" 
# 3. sklearn: OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe= OneHotEncoder()
y = y.reshape(581012, 1)        # (581012, 1) 형태로 변경 (데이터의 내용과 순서만 바뀌지 않으면 됨)
y = ohe.fit_transform(y)
# y = ohe.fit(y)                # 원-핫 인코더는 2차원의 데이터를 받음
# y = ohe.transform(y)
y = y.toarray()                 # numpy 형태로 변경

print(y[:15])                   # 데이터 15개 찍어보기
print(type(y))                  # <class 'numpy.ndarray'>
print(y.shape)                  # (581012, 7)

# 사이킷런에서 원-핫 인코더 쓰고 난 후, numpy 형태(to_array)로 변경
#    원-핫 인코딩 하기 전에 reshape로 형태 변경 후,
#    원-핫 인코더(fit, transform) 이용해서 numpy 형태로 변경 """



################ 사이킷런 OneHotEncoder #################
print(y.shape)
y = y.reshape(581012, 1)
print(y.shape)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
# ohe.fit(y)
# y = ohe.transform(y)
y = ohe.fit_transform(y) # fit과 transform을 한 번에. 위의 두 줄과 같은 코드
y = y.toarray()
# print(type(y))
##########################################################

print(y)
print(y.shape)  # (581012, 54) (581012,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    test_size=0.2,
    random_state=333,
    stratify=y)

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
model.add(Dense(5, activation='relu', input_shape=(54,)))
model.add(Dense(42, activation='sigmoid'))
model.add(Dense(35, activation='relu'))
model.add(Dense(21, activation='linear'))
model.add(Dense(7, activation='softmax'))"""

#2. 모델 구성 (함수형)  # 순차형과 반대로 레이어 구성
input1 = Input(shape=(54,))     # 입력 데이터의 크기(shape)를 Input() 함수의 인자로 입력층 정의
dense1 = Dense(5, activation='linear')(input1)      # 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당
dense2 = Dense(42, activation='sigmoid')(dense1)
dense3 = Dense(35, activation='relu')(dense2)
dense4 = Dense(21, activation='linear')(dense3)
output1 = Dense(7, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)   # 순차형과 달리 model 형태를 마지막에 정의.     Model() 함수에 입력과 출력 정의
model.summary()


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=20,
                              restore_best_weights=True,
                              verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중 분류: loss='categorical_crossentropy'
              metrics=['accuracy'])

import time
start = time.time()

model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          verbose=1)

end = time.time()
print("걸린 시간 : ", end - start)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict[:20])

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : " , y_test[:20])

acc = accuracy_score(y_test, y_predict)
print(acc)


""" 걸린 시간 :  792.5350534915924
loss :  10.165313720703125
accuracy :  0.4876036047935486
y_pred(예측값) :  [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
y_test(원래값) :  [1 6 4 1 1 1 1 1 1 1 0 1 1 0 0 0 1 1 2 0]
0.48760359026875383 """