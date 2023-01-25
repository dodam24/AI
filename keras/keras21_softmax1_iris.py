from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_iris()
print(datasets.DESCR)   # pandas.describe() / pandas.info()
print(datasets.feature_names)   # pandas.columns

x = datasets.data
y = datasets['target']
print(x)
print(y)
print(x.shape, y.shape)   # (150, 4), (150,)

from tensorflow.keras.utils import to_categorical   # one hot encoding 통해서 y.shape = (150, 3)으로 변경됨
y = to_categorical(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,     # shuffle=False로 설정할 경우, 무작위가 아닌 순차적으로 추출 (시계열 데이터 등 순서 유지가 필요한 경우에 사용)
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
model.add(Dense(3, activation='softmax'))
# 소프트맥스(softmax): 입력받은 값을 출력으로 0~1 사이의 값으로 모두 정규화하며, 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수

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
y_predict = np.argmax(y_predict, axis=1)    # array와 비슷한 형태(리스트 등 포함)의 input을 넣어주면 가장 큰 원소의 인덱스 반환
                                            # if a = [3,2,5,4,5]
                                            #    np.argmax(a) = 2가 출력됨 (5의 인덱스) 
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : " , y_test)

acc = accuracy_score(y_test, y_predict)
print(acc)

""" 
시그모이드: 맞냐/아니냐를 뜻하는 이진 분류 문제에 사용
            1이면 맞고 0이면 아님
소프트맥스: 모델의 출력값을 0과 1 사이의 값으로 출력하는 것은 시그모이드와 동일
            하지만 k1에 0.1, k2에 0.2, k3에 0.7 이런 식으로 확률값들의 총합은 항상 1임 
"""
                     
""" 
크로스 엔트로피: Softmax나 Sigmoid를 사용하여 모델의 출력값을 확률값으로 바꿔주면 cross entropy를 이용하여 loss 계산 가능
                머신러닝에서 cross entropy를 사용하여 '예측값'과 '실제값'의 차이 계산
                ex. 다중 분류 문제에서 정답 데이터 y는 [0, 0, 1]과 같이 원핫인코딩 된 벡터
                이는 개, 고양이, 오리 중 오리일 확률이 100%라는 뜻
                모델의 예측값은 [0.2, 0.1, 0.7]와 같은 형태 (총합=1)
                두 확률의 차이를 cross entropy로 계산하면 0.35 도출
                모델의 학습이란 cross entropy를 loss 값으로 삼고 최소화하도록 가중치를 갱신하는 것 
"""

""" 
원핫인코딩: 단 하나의 값만 True, 나머지는 모두 False인 인코딩
            if x=[0,0,0,0,1]에서 원핫인코딩된 데이터 중 가장 큰 값은 1이고, 해당 인덱스를 리턴하면 4 출력
            Softmax를 통해 나온 결과 중 최댓값의 인덱스를 얻고자 할 때 사용
"""