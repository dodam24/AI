import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print(y)
print(np.unique(y)) # [0 1 2]
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

from tensorflow.keras.utils import to_categorical   # (178, 3)으로 변경
y = to_categorical(y)
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=333,
    test_size=0.2,
    stratify=y
)

#2. 모델
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(13,)))
model.add(Dense(42, activation='sigmoid'))
model.add(Dense(35, activation='relu'))
model.add(Dense(21, activation='linear'))
model.add(Dense(3, activation='softmax'))   # 다중 분류: activation='softmax'

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중 분류: loss='categorical_crossentropy'
              metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2,
          verbose=1,
          callbacks=[earlyStopping])

#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

from sklearn.metrics import accuracy_score
import numpy as np

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)

y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : " , y_test)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)