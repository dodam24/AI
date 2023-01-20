from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


# matplotlib 폰트 설정 -한글 폰트 깨짐 처리
from matplotlib import font_manager, rc     

font_path = 'C:/Windows/Fonts/malgun.ttf'   # 폰트 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()     # 폰트 이름 가져오기
rc('font', family=font_name)    # rc 함수를 이용해 폰트의 설정 변경 가능                            

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'     # rcParams 설정을 통해 기본값 설정
matplotlib.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 깨짐 방지


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2) 

#2. 모델 구성
model = Sequential()
# model.add(Dense(5, input_dim=13))         # input_dim은 행과 열
model.add(Dense(5, input_shape=(13,)))      # 다차원일 경우, input_shape 사용
model.add(Dense(40))
model.add(Dense(3))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=300, batch_size=1, 
          validation_split=0.2, 
          verbose=1)    # verbose=1 (default)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("==================================================")
print(hist) # <keras.callbacks.History object at 0x0000017442ACECA0>
print("==================================================")
print(hist.history)
print("==================================================")
print(hist.history['loss'])

# matplotlib 그래프 설정
plt.figure(figsize=(9,6))   # 기본 크기 지정
plt.plot(hist.history['loss'], c='red', 
         marker='.', label='loss')      # plt.plot(): x, y 지점에 선 긋기
plt.plot(hist.history['val_loss'], c='blue', 
         marker='.', label='val_loss')
plt.grid()      # 격자 설정
plt.xlabel('epochs')    # x축 레이블 설정
plt.ylabel('loss')      # y축 레이블 설정
# plt.title('boston loss')
plt.title('보스톤 손실함수')    # 한글 폰트 깨짐 설정해서 문제 해결할 것!
plt.legend()    # 범례 표시
# plt.legeng(loc='upper right')
plt.show()


