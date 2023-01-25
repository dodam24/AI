from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2)

#2. 모델 구성
model = Sequential()
model.add(Dense(50, activation='linear', input_shape=(30,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 이진 분류: activation = 'sigmoid'로 고정

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',     # 이진 분류: loss = 'binary_crossentropy'로 고정
              metrics=['accuracy'])                             # metrics 추가 시, hist의 history에도 accuracy 정보 추가

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
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy) 

y_predict = model.predict(x_test)   # sigmoid 함수 통과 후의 값
y_predict = y_predict > 0.5

print(y_predict)    # [9.7433567e-01]: 실수 값으로 출력됨 -> 정수형으로 변환
print(y_test)       # [1 0 1 1 0 1 1 1 0 1]: 정수

from sklearn.metrics import r2_score, accuracy_score    # accuracy score 추가
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)

""" Epoch 00056: early stopping
loss :  [0.17204582691192627, 0.9385964870452881]   # 값이 2개인 이유: [loss값, metrics 즉, accuracy의 지표] """

""" 이진 분류 (Binary Classification):
결과가 1 또는 0으로 제한
특정 데이터가 1 또는 0으로 분류하는 기준인 Classification thresold는 0.5 값 사용
(확률 값이 0.5 이상이면 1, 0.5 이하면 0) """