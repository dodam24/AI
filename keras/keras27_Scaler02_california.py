import numpy as np
import sklearn as sk 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x)
print(x.shape)   # (20640, 8)   
print(y)
print(y.shape)   # (20640, 1)

print(dataset.feature_names)

print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123)

scaler = MinMaxScaler()                     # minmaxscaler 정의
# scaler = StandardScaler()
scaler.fit(x_train)                         # x값의 범위만큼 가중치 생성
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_test)
x_test = scaler.transform(x_test)           # x_train fit한 가중치 값 범위에 맞춰서 x_test 데이터 변환

#2. 모델 구성
model=Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(75))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',
                              patience=10, 
                              restore_best_weights=True,
                              verbose=1)
hist = model.fit(x_train, y_train, epochs=300, batch_size=25,   
          validation_split=0.2, callbacks=[EarlyStopping], 
          verbose=1)

""" 
신경망 모델의 훈련에서 사용되는 fit() 메서드는 History 객체를 반환
History.history 속성: 훈련 과정에서 에포크(epoch)에 따른 정확도(accuracy)와 같은 지표와 손실값 기록
                    또한, 검증(validation)의 지표와 손실값도 기록 """
                   
                    
""" 
validation_split: 0에서 1 사이의 값으로,
                0.25로 설정해주면 훈련(train) 데이터의 25%를 검증에 사용
                if 60,000개의 데이터 중 45,000개를 훈련에 사용하고 15,000개를 검증에 사용 
 """

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


print("==================================================")
print(hist)     # <keras.callbacks.History object at 0x0000017442ACECA0>
print("==================================================")
print(hist.history)     # loss의 변화량 값을 dictionary 형태로 출력. (key:values) 즉, 키-값의 쌍형태. value: 리스트형     
print("==================================================")
print(hist.history['loss'])     # loss값만 출력


y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', 
         marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', 
         marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('boston loss')
plt.legend()
# plt.legeng(loc='upper right')
plt.show()


""" 
Epoch 00012: early stopping
loss :  [0.5558849573135376, 0.529268205165863]
==================================================
<keras.callbacks.History object at 0x00000240385C6C70>
==================================================
{'loss': [0.9910679459571838, 0.5822520852088928, 0.5606656670570374, 0.5524595975875854, 0.5461922287940979, 0.5498453974723816, 0.5506868362426758, 0.5428751111030579, 0.5434556007385254, 0.5395942330360413, 0.5405368804931641, 0.54343181848526], 
'mae': [0.7462883591651917, 0.5616880655288696, 0.5533575415611267, 0.5500189065933228, 0.5472128391265869, 0.5515275597572327, 0.5500151515007019, 0.5454432368278503, 0.5449839234352112, 0.5432299971580505, 0.5451372265815735, 0.5451323986053467], 
'val_loss': [0.6011999845504761, 0.5680411458015442, 0.571496307849884, 0.6106994152069092, 0.6220614910125732, 0.6366430521011353, 0.6562101244926453, 0.6533588767051697, 0.6591547131538391, 0.6827123761177063, 0.7126474976539612, 0.6989474296569824], 
'val_mae': [0.549619734287262, 0.5313434600830078, 0.5367369651794434, 0.5744274258613586, 0.5333132743835449, 0.537380576133728, 0.5278118848800659, 0.5597835183143616, 0.5457961559295654, 0.5496063828468323, 0.5289499759674072, 0.5385361909866333]}
==================================================
[0.9910679459571838, 0.5822520852088928, 0.5606656670570374, 0.5524595975875854, 0.5461922287940979, 0.5498453974723816, 0.5506868362426758, 0.5428751111030579, 0.5434556007385254, 0.5395942330360413, 0.5405368804931641, 0.54343181848526]
RMSE :  0.7455769712344344
R2 :  0.5796040180188027 """
