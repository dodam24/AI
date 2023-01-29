import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

#1. 데이터
datasets = fetch_covtype()      # 수종 예측 데이터
x = datasets.data
y = datasets.target
print(x.shape, y.shape)         # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
""" 
(array([1, 2, 3, 4, 5, 6, 7]), 
array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64)) """

# 원-핫 인코딩
y = to_categorical(y)
print(y.shape)                  # (581012, 8)

y = np.delete(y, 0, axis=1)     # 0번째 열 삭제 (axis=0은 행, axis=1은 열 기준)
print(y.shape)                  # (581012, 7)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2, stratify=y)

# 스케일링
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)      # (464809, 54) (116203, 54)

x_train = x_train.reshape(464809, 54, 1)
x_test = x_test.reshape(116203, 54, 1)


#2. 모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(54, 1), padding='same'))
model.add(Conv1D(64, 2, input_shape=(54, 1), padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.summary()


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              patience=10,
                              restore_best_weights=True,
                              verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=25,
                 validation_split=0.2, callbacks=[EarlyStopping], verbose=1)


#4. 평가, 예측
from sklearn.metrics import accuracy_score
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_test = np.argmax(y_test, axis=1)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

accuracy = accuracy_score(y_test, y_predict)

print('accuracy : ', accuracy)



""" Epoch 00056: early stopping
3632/3632 [==============================] - 2s 643us/step - loss: 0.4006
loss :  0.40055522322654724
accuracy :  0.8355292032047366 """