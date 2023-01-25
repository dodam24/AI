from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape)   # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2)

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
model.add(Dense(50, activation='linear', input_shape=(30,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 이진 분류 """

#2. 모델 구성(함수형)   # 순차형과 반대로 레이어 구성
input1 = Input(shape=(30,))     # 입력 데이터의 크기(shape)를 Input() 함수의 인자로 입력층 정의
dense1 = Dense(50, activation='linear')(input1)     # 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(10, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)       # 순차형과 달리 model 형태를 마지막에 정의.     Model() 함수에 입력과 출력 정의
model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',   # 이진 분류 (Binary_crossentropy)
              metrics=['accuracy'])  

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',
                              patience=20, 
                              restore_best_weights=True,
                              verbose=1) 
model.fit(x_train, y_train, epochs=10000, batch_size=16,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss, accuracy : ', loss)
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy) 

y_predict = model.predict(x_test) 
y_predict = y_predict > 0.5         # 0.5 이상이면 True 반환, 0.5 이하이면 False 반환

print(y_predict)    # [9.7433567e-01]: 실수 값으로 출력됨 -> 정수형으로 변환
print(y_test)       # [1 0 1 1 0 1 1 1 0 1]: 정수

from sklearn.metrics import r2_score, accuracy_score
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)


""" 
Epoch 00078: early stopping
4/4 [==============================] - 0s 335us/step - loss: 0.1966 - accuracy: 0.9649
loss :  0.19662971794605255
accuracy :  0.9649122953414917 """

# Epoch 00056: early stopping
# loss :  [0.17204582691192627, 0.9385964870452881]   # [loss값, metrics 즉, accuracy의 지표]