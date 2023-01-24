import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#1. 데이터
datasets = fetch_covtype()      # 대표 수종 데이터 (미국 삼림의 각 영역별 특징으로부터 대표적인 나무의 종류 예측)
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)         # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
""" (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
dtype=int64)) """
                                
                                
# 원-핫 인코딩 방법 3가지       
# from tensorflow.keras.utils import to_categorical        
# y = to_categorical(y)   # one hot encoding

import pandas as pd                         # pd.get_dummies 처리: 결측값 제외하고 0과 1로 구성된 더미값을 생성             
y = pd.get_dummies(y, dummy_na=True)        # 결측값 처리(dummy_na=True 옵션): Nan을 생성하여 결측값도 인코딩 처리함
                                            #   dummy_na=True: (581012, 8) // dummy_na=Flase: (581012, 7)
                                            
                                            # numpy 자료형이 pandas 자료형을 바로 받아들이지 못하기 때문에 마지막에 에러 발생
                                            #   (np.argmax 부분에서 y_test가 pandas 형태이므로)
                                            # numpy: 배열 내의 모든 값의 자료형이 같아야 함 (숫자형 자료: int, uint, float, complex, bool)
                                            # pandas: 다양한 자료형을 담을 수 있음 (데이터프레임 형태)
                                            #   따라서 np.argmax 대신 tf.argmax 사용해야 함

y = y.values                                # values 쓰거나 .numpy() 쓰면 오류 해결 (pandas 데이터를 numpy 데이터로 바꿔주는 과정)    

""" 
from sklearn.preprocessing import OneHotEncoder           
ohe = OneHotEncoder()
# shape를 맞추는 작업
y = ohe.fit_transform(y) 
"""

print(y)
print(type(y))      # <class 'numpy.ndarray'>
print(y.shape)      # (581012, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2,stratify=y)

#2. 모델
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(54,)))
model.add(Dense(42, activation='sigmoid'))
model.add(Dense(35, activation='relu'))
model.add(Dense(21, activation='linear'))
model.add(Dense(8, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=20,
                              restore_best_weights=True,        # True로 설정해야 종료 시점이 아닌, early stopping 지점의 최적의 weight값 사용 가능
                              verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중 분류: loss='categorical_crossentropy'
              metrics=['accuracy'])

import time
start = time.time()

model.fit(x_train, y_train, epochs=3000, batch_size=32,
          validation_split=0.2,
          verbose=1,
          callbacks=[earlyStopping])

end = time.time()
print("걸린 시간 : ", end - start)

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)


y_predict = model.predict(x_test)               # 예측값
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test, axis=1)              # 원래 값
print("y_test(원래값) : " , y_test)

acc = accuracy_score(y_test, y_predict)         # 정확도 (예측값과 원래 값의 차이)
print(acc)

print('time : ', end - start)