
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 그래프 한글 깨짐 방지
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'   # 저장된 경로에서 폰트 불러오기
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)   # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2) 

#2. 모델 구성
model = Sequential()
# model.add(Dense(5, input_dim=13))         # input_dim은 행과 열일 때
model.add(Dense(5, input_shape=(13,)))      # 다차원: input_shape 사용 (13, )
                                            # if (100, 10, 5)일 때, (10, 5)로 표현 가능. 맨 앞의 100은 데이터 개수
model.add(Dense(40000))
model.add(Dense(3))
model.add(Dense(20000))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')


from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss',               # history의 val_loss 최솟값 이용 
                              mode='min',                       # accuracy 사용 시에는 max로 설정 (정확도가 높을수록 좋기 때문에)
                              patience=10, 
                   
                              restore_best_weights=True,
                              verbose=1)

hist = model.fit(x_train, y_train, epochs=300, batch_size=1,    # training이 끝난 학습 결과를 history에 저장
          validation_split=0.2, callbacks=[EarlyStopping],      # val_loss 기준으로 최솟값이 n번 이상 갱신 안 되면 훈련 중지
          verbose=1)    # verbose=1 (default)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("==================================================")
print(hist)     # <keras.callbacks.History object at 0x0000017442ACECA0>
print("==================================================")
print(hist.history)     # loss 변화량 값을 dictionary 형태로 출력 (key:value) 키-값 쌍형태. value: 리스트형
print("==================================================")
print(hist.history['loss'])     # loss 값만 출력

#5. 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))   # 그래프 사이즈 설정
plt.plot(hist.history['loss'], c='red',                
         marker='.', label='loss')                  
plt.plot(hist.history['val_loss'], c='blue', 
         marker='.', label='val_loss')
plt.grid()  # 격자 무늬
plt.xlabel('epochs')    # x축 이름
plt.ylabel('loss')      # y축 이름
plt.title('boston loss')
plt.legend()
# plt.legend(loc='upper right')
plt.show()

""" 결과
Epoch 00025: early stopping
4/4 [==============================] - 0s 2ms/step - loss: 41.9196
loss :  41.919586181640625 """


""" EarlyStopping 콜백 함수를 활용하여, model의 성능 지표가 설정한 epoch동안 개선되지 않을 때 조기 종료 실행
+ ModelCheckpoint로부터 가장 best model을 다시 로드하여 학습 재개 """