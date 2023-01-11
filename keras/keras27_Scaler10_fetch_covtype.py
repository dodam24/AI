import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))

# ########## keras to_categorical ##########
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)   # one hot encoding
# print(y.shape)   # (58102, 8)
# print(type(y)) # <class 'numpy.ndarray'>
# print(y[:10])
# print(np.unique(y[:,0], return_counts=True))
# # (array([0.], dtype=float32), array([581012], dtype=int64))
# print(np.unique(y[:,1], return_counts=True))
# print("===================================")
# y = np.delete(y, 0, axis=1) # axis=1(열), axis=0(행)
# print(y.shape)
# print(y[:10])
# print(np.unique(y[:,0], return_counts=True))
# ############################################


############# pandas get_dummies #############
# import pandas as pd
# y = pd.get_dummies(y)
# print(y[:10])
# print(type(y)) # 
# # y = y.values
# y = y.to_numpy()
# print(type(y))
# print(y.shape)

# 판다스는 헤더와 인덱스가 존재함
# 인덱스와 헤더(컬럼명)는 연산에 사용되지 않음
##############################################


# ############## sklearn OneHotEncoder #############
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder() # 정의
# # y = ohe.fit_transform()  # shape를 맞추는 작업
# y = ohe.fit(y)
# # onehotencoder는 2차원의 데이터를 받음
# y = y.reshape() # (581012, 1) 형태로 변경 (데이터의 내용과 순서만 바뀌지 않으면 됨)
# y = ohe.transform(y)
# y = y.toarray()

# print(y[:15])
# print(type(y))
# print(y.shape)
# 사이킷런에서 원핫인코더 쓰고 난 후, to array(numpy 형태)로 바꿔준다.
# 원핫인코더 하기 전에 reshape로 형태 변경 후, 원핫인코더(fit, transform) 이용해서 to array(numpy 형태)로 바꿔준다.
###################################################


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
#print(type(y))
##########################################################


print(y)
print(y.shape)  # (581012, 54) (581012,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    test_size=0.2,
    random_state=333,
    stratify=y)

scaler = MinMaxScaler() # MinMaxscaler 정의
# stdandardScler = Staler()
scaler.fit(x_train) # x값의 범위만큼의 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test) # 시작 (transform해야 바뀐다.)

#2. 모델
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(54,)))
model.add(Dense(42, activation='sigmoid'))
model.add(Dense(35, activation='relu'))
model.add(Dense(21, activation='linear'))
model.add(Dense(7, activation='softmax'))

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