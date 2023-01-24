from sklearn.datasets import load_iris  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_iris()          # 꽃잎의 4가지 특성(꽃잎 길이, 꽃받침 너비 등)으로 붓꽃의 종류(3가지) 분류
print(datasets.DESCR)           # pandas.describe() / pandas.info()
print(datasets.feature_names)   # pandas.columns

x = datasets.data
y = datasets['target']

print(x)
"""
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]
 [5.4 3.9 1.7 0.4] 
 .
 .
 .
 ]]
 """

print(y)
""" 
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0        
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1        
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2        
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2        
 2 2]
 """
 
print(x.shape, y.shape)     # (150, 4), (150,)


# from tensorflow.keras.utils import to_categorical     # one hot encoding을 통해서 y.shape = (150, 3)으로 변경됨
# y = to_categorical(y)
# print(y)
# print(y.shape)    # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=333, 
    test_size=0.2,
    stratify=y
)
# print(y_train)
# print(y_test)

#2. 모델
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(4,)))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(3, activation='relu'))
model.add(Dense(2, activation='linear'))
model.add(Dense(3, activation='softmax'))   # y의 클래스 개수 

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',   # 0부터 시작해야 sparse_categorical_crossentropy 적용 가능
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=1,
          validation_split=0.2,
          verbose=1)

""" 
훈련 데이터의 label(y, target) 값이 원-핫 벡터 형태인 경우: categorical_crossentropy 사용
    ex. y_true = [[0, 1, 0][0, 0, 1]]
훈련 데이터의 label(y, target) 값이 정수(int) 형태인 경우: sparse_categorical_crossentropy 사용
    ex. y_true = [1, 2]
"""
    

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
# y_test = np.argmax(y_test, axis=1)    # 원-핫 인코딩 안 했으니까 여기서 argmax를 사용할 필요가 없음
print("y_test(원래값) : " , y_test)
# acc = accuracy_score(y_test, y_predict)
# print(acc)


""" 원-핫 인코딩 하는 방법 (3가지)
1. keras의 to_categorical
2. pandas의 get_dummies
3. sklearn의 OneHotEncoder """


"""
# #################### keras to_categorical ####################
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)       # one hot encoding
print(y.shape)              # (58102, 8)
print(type(y))              # <class 'numpy.ndarray'>
print(y[:10])
print(np.unique(y[:,0], return_counts=True))
    # (array([0.], dtype=float32), array([581012], dtype=int64))
print(np.unique(y[:,1], return_counts=True))

print("===================================")
y = np.delete(y, 0, axis=1) # axis=1(열), axis=0(행)
print(y.shape)
print(y[:10])
print(np.unique(y[:,0], return_counts=True))
# ############################################################### """


"""
# #################### pandas get_dummies ####################
import pandas as pd
y = pd.get_dummies(y)
print(y[:10])
print(type(y))
# y = y.values
y = y.to_numpy()
print(type(y))
print(y.shape)

# 판다스는 헤더와 인덱스가 존재함
# 인덱스와 헤더(컬럼명)는 연산에 사용되지 않음
# ############################################################# """


"""
# #################### sklearn OneHotEncoder ####################
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()           # 원-핫 인코더에 대한 정의
y = ohe.fit_transform()         # shape를 맞추는 작업
    # y = ohe.fit(y)            # OneHotEncoder는 2차원의 데이터를 받음
y = y.reshape()                 # (581012, 1) 형태로 변경 (데이터의 내용과 순서만 바뀌지 않으면 됨)
    # y = ohe.transform(y)
y = y.toarray()

# print(y[:15])
# print(type(y))
# print(y.shape)
# 사이킷런에서 원-핫인코더 쓰고 난 후, to array(numpy 형태)로 바꿔준다.
# 원-핫 인코딩 하기 전에 reshape로 형태 변경 후, 원-핫 인코더(fit, transform) 이용해서 to array(numpy) 형태로 바꿔준다.
# ################################################################ """


"""
# #################### 사이킷런 OneHotEncoder ####################
print(y.shape)
y = y.reshape(581012, 1)    # 원-핫 인코딩 전에 reshape로 형태 변경
print(y.shape)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform           # fit과 transform을 한 번에 수행. 아래 두 줄과 같은 코드
    # ohe.fit(y)                # 원-핫 인코딩: to array(numpy) 형태로 변경  
    # y = ohe.transform(y)
y = y.toarray()                 # numpy 형태로 변경
print(type(y))

# print(y[:15])
# print(type(y))
# print(y.shape)
# ################################################################# """

"""
loss :  0.13718558847904205
accuracy :  0.9333333373069763 """