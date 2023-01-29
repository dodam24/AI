# DNN 적용 (Flatten으로 4차원 모델을 2차원으로 변경, reshape)

import numpy as np
from tensorflow.keras.datasets import mnist

# 경로 설정
# path = 'C:/study/_save/MCP/'
filepath = './_save/MCP/'
filename = '{epoch: 04d}-{val_loss: .4f}.hdf5'

# 파일 이름 설정 (날짜)
import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")   # 0114_1844

print(date)
print(type(date))                   # <class 'str'>
  
  
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)  -> input_shape = (28, 28, 1)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

''' x_train = x_train.reshape(60000, 28, 28, 1)     # 4차원으로 변경
x_test = x_test.reshape(10000, 28, 28, 1) '''

x_train = x_train.reshape(60000, 28*28, 1)     # 2차원으로 변경
# x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 28*28, 1)
# x_test = x_test.reshape(10000, 784)

# min_max scaling (나누기 255)      # 숫자가 너무 커지기 때문에
x_train = x_train/255
x_test = x_test/255

print(np.unique(y_train, return_counts=True))   # 배열 내 고유한 원소별 개수
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#2. 모델                    # padding 추가, maxpooling 추가, strides 추가
model = Sequential()
model.add(LSTM(128, input_shape=(784, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 손글씨 이미지 분류 (숫자 0~9)
                                            # output 노드 10개이므로 다중 분류에 해당

# input_shape의 값에서 kernel_size 값을 빼준 후, +1 = 결과 값.  ex) (28, 28, 1) - (3, 3) + 1 = (26, 26, 128)

model.summary()


#3. 컴파일, 훈련
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])


# earlystopping 설정
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
# patience=10: 10회 이상 모델 성능에 차이가 없으면 학습 중단!

# modelcheckpoint 설정
mcp = ModelCheckpoint(monitor='val_loss', model='auto', verbose=1, save_best_only=True,
                      filepath = filepath + 'k34_mnist_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=32, validation_split=0.25, callbacks=[es, mcp])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
print('val_acc : ', results[3])



""" 
MNIST를 이용한 손글씨 인식하기
 - mnist 데이터는 케라스를 이용해서 호출
 - 고등학생과 인구조사국 직원 등이 쓴 손글씨를 이용해 만든 데이터로 구성
 - 70,000개의 글자 이미지에 각각 0부터 9까지의 이름표를 붙인 데이터셋
 - 총 70,000개의 이미지 중 60,000개를 학습용으로, 10,000개를 테스트용으로 미리 구분해 놓음
 
 1) 데이터 전처리
 plt.imshow(x_train[0], 'gray'): 이미지 출력
 - imshow(): 이미지 출력
 - x_train[0]: 첫 번째 이미지
 - 'gray': 흑백 이미지 출력
 
 데이터
 x : 이미지 데이터 (x_train, x_test)
 y : 0부터 9까지의 이름표 (y_train, y_test)
  -> 784개의 속성을 이용해 10개 클래스 중 하나를 맞추는 문제
  
가로: 28, 세로:28 = 총 784개의 픽셀로 구성
각 픽셀의 밝기: 0 ~ 255
2차원 배열(28 X 28 행렬)에서 1차원 배열로 변경해줘야 함
 - reshape(총 샘플 수, 1차원 속성의 수)

데이터 정규화: 데이터 폭이 클 때, 적절한 값으로 분산의 정도를 바꿈
 - 현재 값인 0 ~ 255를 0 ~ 1 사이의 값으로 변경
 - asstype(float) 후, 255로 나눔
 
 원-핫 인코딩: 딥러닝 분류 문제 해결
 - 0 ~ 9 까지의 클래스를 0과 1로 이루어진 벡터로 변경
 - to_categorical(클래스, 클래스 개수)
 
 

2) 딥러닝 기본 프레임 만들기
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
- 은닉층의 활성화 함수: relu, 출력층의 활성화 함수: softmax

model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32, validation_split=0.25, callbacks=[es, mcp])
 - 샘플 32개를 100번 실행(?)



3) 컨볼루션 신경망(CNN)
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32,32,3), activation='relu'))
# Conv2D(숫자, kernel_size, input_shape, activation): 케라스에 컨볼루션 층을 추가하는 함수
 - 첫 번째 인자: 마스크를 몇 개 적용할 지 결정 (filters)
 - kernel_size: 마스크(커널)의 크기 결정 (2X2 사이즈의 마스크 사용)
 - input_shape=(행,열, 색상 또는 흑백): 맨 처음 층에 입력되는 값 알려주기
                입력 이미지가 색상인 경우 3, 흑백인 경우 1 선택
                
model.add(Conv2D(filters=64, kernel_size=(2,2)))
# 마스크 64개를 적용한 컨볼루션 층 추가 

 - Padding의 종류
    1. Valid Padding : padding 하지 않는 것을 의미
    2. Same Padding : output image가 input image의 크기와 동일 한 것 
 
 
4) 맥스 풀링
 특성맵이 절반의 크기로 다운샘플링. 맥스 풀링은 커널과 겹치는 영역 안에서 최대값을 추출하는 방식으로 다운샘플링


5) 드롭아웃(Drop-out), 플래튼(Flatten)
 드롭아웃(Drop-out): 과적합(overfitting) 방지 목적
 플래튼(Flatten): 2차원 배열을 1차원 배열로 바꿈
 - 컨볼루션 층, 맥스풀링 층: 2차원 배열
 - 입력층: 1차원 배열 (활성화 함수 적용 시, 필요함)
 

"""
