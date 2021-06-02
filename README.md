# MJU-Artificial-intelligence1-DogePredict

## 1. 개요

명지대학교 4학년 1학기 인공지능1 강의 기말프로젝트로 최근에 가장 변동성이 심했던 가상화폐인 도지코인의 종가 데이터로 가격을 예측하는 학습 시스템을 구현해보았다.

상당히 간략한 코드이고 외부 변수에 따라 변동성이 큰 점은 배제하였기 때문에 믿을만하지는 않다.

### 1.1. 사용 라이브러리
- LSTM - 시계열 데이터 처리 모델
- padas, numpy - 데이터 분석 패키지
- request - api요청용
- datetime, timedelta - 시간 데이터 처리 및 현재 시간 불러오기용
- Sequential - 케라스 학습 처리 모델
- Dense, Dropout - 성능 향상용 은닉층

### 1.2. 사용 툴
- Google Colab - 학습이 꽤 빠르게 진행되서 신세계를 경험할 수 있음

## 2. 상세코드

### 2.1. 패키지 임포트
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import matplotlib.ticker as ticker
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from datetime import datetime, timedelta
```

### 2.2. 데이터 불러오기
```python
arr = []
for i in range(2):
    today = datetime.today() - timedelta(i * 30)  #6시간 간격 2달치 데이터를 가져오기 위해서 api요청으로 180번(30일) * 2번을 요청한다.
    req = requests.get('https://crix-api-endpoint.upbit.com/v1/crix/candles/minutes/240?code=CRIX.UPBIT.KRW-DOGE&count=180&to='+ today.strftime("%Y-%m-%d")+'%2024:00:00')
    arr += req.json()        

#json 데이터화
arr.reverse()                                     #최신 데이터부터 먼저 들어오기 때문에 순서를 역순으로 바꿔준다.
df = pd.DataFrame(arr)                            #pandas.dataFrame으로 가공하기 편한 데이터로 만들어준다.
print(df)
```

### 2.3. 데이터 가공
```python
#데이터 자르기
df['candleDateTimeKst'] = df['candleDateTimeKst'].str.slice(start=0,stop=19).to_numpy() #string으로 넘어온 시간 데이터를 보기 편하게 잘라준다.
datachart = df['tradePrice']                                                            #y축에 사용할 수 있도록 가격데이터를 뽑아낸다. 추가적인 가공이 필요하다.
time = df['candleDateTimeKst'].to_numpy()                                               #x축에 사용할 수 있도록 시간데이터도 뽑아내준다.
length = len(datachart)-1


plt.figure(figsize=(24, 16))                                                            #화면 크기 지정, 실제 데이터의 형태를 그래프로 확인한다.
plt.plot(time, datachart)
ax=plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(60))                                  #X축에 사용될 시간 데이터의 간격을 설정해줬다.
plt.xlabel('time')
plt.ylabel('price')
```

### 2.4. window_size 설정 후 학습데이터와 결과데이터로 나누기
```python
window_size = 6

def seq2dataset(datachart,window_size):                   #Window사이즈를 기준으로 학습데이터와 결과데이터를 나눔
    X = []; Y = []
    for i in range(len(datachart) - window_size):
        X.append(np.array(datachart.iloc[i:i+window_size]))
        Y.append(np.array(datachart.iloc[i+window_size]))
    return np.array(X), np.array(Y)

X, Y = seq2dataset(datachart,window_size);
X = X.reshape((X.shape[0], X.shape[1], 1))                #LSTM에 사용하기 위해서는 데이터 형태가 3차원 데이터를 사용해야 한다고 함
```

### 2.5. 학습모델 구축 후 예측 결과 확인
```python
split = int(len(X)*0.7)                           #학습데이터(70%)와 테스트데이터(30%)로  나눔
x_train=X[0:split]; y_train=Y[0:split]
x_test=X[split:]; y_test=Y[split:]

model=Sequential()                                #학습모델 구축
model.add(LSTM(128,activation='relu',input_shape=(x_train.shape[1], x_train.shape[2], ))) #활성함수로는 relu 사용
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['mae']) #mae를 사용해야 특이값에 덜 민감하다.
model.fit(x_train, y_train, epochs=30,batch_size=32,validation_data=(x_test,y_test),verbose=2)
model.summary()

#예측
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)


#실제 데이터
plt.figure(figsize=(24, 16))
plt.plot(time, datachart)

#훈련 결과
split_tp = split + window_size
plt.plot(np.arange(window_size, split_tp, 1), train_predict, color='g')

#테스트 결과
plt.plot(np.arange(split_tp, split_tp + len(test_predict), 1), test_predict, color='r')

plt.ylabel('Price')
ax=plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
plt.show()

#일부 데이터 확대
x_range = range(length - 15,length + 1)
plt.figure(figsize=(24, 16))
plt.plot(time[x_range], datachart[x_range],color='red')
plt.plot(time[x_range], test_predict[range(len(test_predict) - 16 ,len(test_predict))], color='blue')
ax=plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.legend(['True prices', 'Predicted prices'], loc='best')
plt.show()
```
#### 2.5.1 전체 데이터 예측
![전체 데이터](C:/Users/cksgh/OneDrive/바탕 화면/ai1.png)

### 2.6. 미래 예측
```python
now = time[time.size - 1]
print("최근 시간 데이터 : ", now)

newdata = np.array(datachart.iloc[length + 1 - window_size:length+1])   #미래 예측을 위한 기존 데이터 추출
newdata = newdata.reshape((1, newdata.shape[0], 1))                     #3차원 데이터 변환 데이터 배열이 1개, 특징이 window_size개

newPredict = model.predict(newdata)
print("최근 가격 데이터 : ", datachart[length])
print("4시간 뒤 가격 예상 : ", newPredict)
sub = round(newPredict[0][0] - datachart[length], 2)
rate = round((newPredict[0][0]/datachart[length] - 1) * 100, 2)
if(sub > 0):
  print("예상 시간별 손익 : ", "+"+ str(sub))
else:
  print("예상 시간별 손익 : ", str(sub))
print("예상 수익률 : ", str(rate))
```
