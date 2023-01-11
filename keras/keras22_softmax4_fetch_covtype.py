import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)   # one hot encoding

# import pandas as pd
# y = pd.get_dummies(y)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
# shape를 맞추는 작업
y = ohe.fit_transform(y)

print(y)
print(y.shape)  # (581012, 54) (581012,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    random_state=333,
    test_size=0.2,
    stratify=y
)

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
                              restore_best_weights=True,
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


y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print("y_pred(예측값) : ", y_predict)
y_test = np.argmax(y_test, axis=1)
print("y_test(원래값) : " , y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)
print('time : ', end - start)