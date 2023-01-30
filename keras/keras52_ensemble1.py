# 앙상블 모델: 여러 개의 모델을 합쳐서 하나의 모델을 만드는 것

import numpy as np

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x1_datasets.shape)    # (100, 2)      # 삼성전자 시가, 고가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T
print(x2_datasets.shape)    # (100, 3)      # 아모레 시가, 고가, 종가

y = np.array(range(2001, 2101))   # (100, )       # 삼성전자의 하루 뒤 종가 예측

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.7, random_state=1234
)

print(x1_train.shape, x2_train.shape, y_train.shape)
print(x1_test.shape, x2_test.shape, y_test.shape)


#2-1. 모델 1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

#2-2. 모델 2
input2 = Input(shape=(3,))
dense21 = Dense(21, activation='linear', name='ds21')(input1)
dense22 = Dense(22, activation='linear', name='ds22')(dense1)
output2 = Dense(23, activation='linear', name='ds23')(dense2)

#2-3. 모델 병합
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=8)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)
