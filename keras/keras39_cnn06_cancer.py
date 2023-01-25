from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
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

print(x_train.shape, x_test.shape)      # (455, 30) (114, 30)

x_train = x_train.reshape(455, 5, 2, 3)       
x_test = x_test.reshape(114, 5, 2, 3)
print(x_train.shape, x_test.shape)


#2. 모델 구성 (순차형)
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(5, 2, 3)))
model.add(Flatten())
model.add(Dense(1, activation='linear'))

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
model.fit(x_train, y_train, epochs=10, batch_size=16,
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
y_predict = y_predict > 0.5         # 0.5 이상이면 True 반환, 0.5 이하이면 Flase 반환

print(y_predict)   # [9.7433567e-01]: 실수 값으로 출력됨 -> 정수형으로 변환
print(y_test)      # [1 0 1 1 0 1 1 1 0 1]: 정수
""" 
[1 0 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 1 0 0 0 1 0 1 1 1 1 1 0 0 1 1 1 0 1 1 1
 1 1 0 1 0 0 0 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 0 1 1 0
 1 0 1 0 1 0 1 1 0 0 0 0 1 1 1 0 0 1 1 1 0 1 1 1 1 1 0 0 0 1 1 1 1 1 0 1 1
 1 0 0] """

from sklearn.metrics import r2_score, accuracy_score
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)


""" loss :  10.418604850769043
accuracy :  0.3245614171028137 """