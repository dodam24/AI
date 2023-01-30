import numpy as np

#1. 데이터
x_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x_datasets.shape)    # (100, 2)


y1 = np.array(range(2001, 2101))     # (100, )       # 삼성전자의 하루 뒤 종가 예측
y2 = np.array(range(201, 301))      # (100,)        # 아모레의 하루 뒤 종가

from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x_datasets, y1, y2, train_size=0.7, random_state=1234
)

print(x_train.shape, y1_train.shape, y2_train.shape)
print(x_test.shape, y1_test.shape, y2_test.shape)


#2-1. 모델 1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='ds11')(input1)
dense2 = Dense(10, activation='relu', name='ds12')(dense1)
dense3 = Dense(10, activation='relu', name='ds13')(dense2)
output1 = Dense(10, activation='relu', name='ds14')(dense3)

''' #2-2. 모델 2
input2 = Input(shape=(3,))
dense1 = Dense(20, activation='linear', name='ds21')(input1)
dense2 = Dense(20, activation='linear', name='ds22')(dense1)
output2 = Dense(20, activation='linear', name='ds23')(dense2)

#2-3. 모델 3
input3 = Input(shape=(2,))
dense1 = Dense(30, activation='relu', name='ds31')(input1)
dense2 = Dense(30, activation='relu', name='ds32')(dense1)
output3 = Dense(30, activation='relu', name='ds33')(dense2) '''

''' #2-4. 모델 병합
from tensorflow.keras.layers import Concatenate
merge1 = Concatenate([output1, output2, output3], name='mg1')
merge2 = Dense(10, activation='relu', name='mg2')(merge1)
merge3 = Dense(10, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3) '''

#2-5. 모델 5 = 분기 1 (모델 병합한 것 다시 둘로 쪼개기)
dense5 = Dense(20, activation='relu', name='ds51')(output1)
dense5 = Dense(20, activation='relu', name='ds52')(dense5)
output5 = Dense(20, activation='relu', name='ds54')(dense5)

#2-6. 모델 6 = 분기 2 (모델 병합한 것 다시 둘로 쪼개기)
dense6 = Dense(10, activation='linear', name='ds61')(output1)
dense6 = Dense(10, activation='linear', name='ds62')(dense6)
output6 = Dense(10, activation='linear', name='ds63')(dense6)

model = Model(inputs=input1, outputs=[output5, output6])

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, [y1_train, y2_train], epochs=500, batch_size=5)


#4. 평가, 예측
loss = model.evaluate(x_test, [y1_test, y2_test])
print('loss : ', loss)


# loss :  [2974376.5, 2973780.25, 596.2608032226562]