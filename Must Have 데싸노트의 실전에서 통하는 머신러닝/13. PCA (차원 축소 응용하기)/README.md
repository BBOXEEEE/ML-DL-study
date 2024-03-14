# 13. 주성분 분석 (PCA) : 차원 축소 응용하기

## 🔥 PCA (Principal Component Analysis) 란?

**PCA (Principal Component Analysis, 주성분 분석)** 은 비지도 학습에 속하며 어떤 것을 예측하지도 분류하지도 않는다.
PCA의 목적은 **데이터의 차원을 축소** 하는데 있다.
차원 축소를 간단히 말하면 변수의 개수를 줄이되, 가능한 그 특성을 보존해내는 기법이다. 기존 변수들의 정보를 모두 반영하는 새로운 변수들을 만드는 방식으로 차원 축소를 한다.

- 장점
	- 다차원을 2차원에 적합하도록 차원 축소하여 시각화에 유용하다.
	- 변수 간의 높은 상관관계 문제를 해결한다.
- 단점
	- 기존 변수가 아닌 새로운 변수를 사용하여 해석하는데 어려움이 있다.
	- 차원이 축소됨에 따라 정보 손실이 불가피하다.

## 🔥 차원을 축소해서 그래프 그리기

클러스터링 모델의 예측 결과에 사용한 독립변수가 너무 많으면 그래프 한 장으로 깔끔하게 표현하기 어렵다.
일부 변수만을 표현하면 정보를 왜곡할 수 있다. 이러한 경우 PCA를 사용하면 이 문제를 해결할 수 있다.

### 1. 데이터 불러오기

```python
import pandas as pd

file_url = 'https://raw.githubusercontent.com/musthave-ML10/data_source/main/customer_pca.csv'  
customer = pd.read_csv(file_url)
```

### 2. 데이터셋 확인

```python
print(customer.head())

# 독립 변수, 종속 변수 분리  
customer_X = customer.drop('label', axis=1)  
customer_y = customer['label']
```

- PCA를 이용해 변수의 수가 너무 많아 2차원 그래프로 표현하기 힘든 것을 변수 2개로 차원을 축소해 산점도 그래프로 출력한다.
- 이를 통해 클러스터가 어떻게 나뉘었는지를 확인한다.

### 3. 그래프 표현을 위한 차원 축소

```python
from sklearn.decomposition import PCA  
  
pca = PCA(n_components=2)   # n_components : 주성분 개수  
pca.fit(customer_X)
customer_pca = pca.transform(customer_X)

# 데이터프레임 형태로 변환
customer_pca = pd.DataFrame(customer_pca, columns=['PC1', 'PC2'])  
customer_pca = customer_pca.join(customer_y)  
  
# 최종 데이터 확인  
print(customer_pca.head())  
  
# 산점도 그래프  
sns.scatterplot(x='PC1', y='PC2', data=customer_pca, hue='label', palette='rainbow')  
plt.show()
```

- 현재 그래프를 통해 확인할 수 있는 것은 클러스터들이 얼마나 잘 나뉘었는지를 대략 확인하는 것이다.
- PCA를 통해 얻어낸 변수 `PC1` 과 `PC2` 는 기존의 모든 변수를 복합적으로 반영하여 만들어졌기 때문에 명료하게 해석하기 쉽지 않다.

추가적으로 각 주성분과 기존 변수와의 상관관계를 알 수 있다.

```python
df_comp = pd.DataFrame(pca.components_, columns=customer_X.columns)

# 히트맵
sns.heatmap(df_comp, cmap='coolwarm')  
plt.show()
```

## 🔥 속도와 예측력 향상시키기

차원을 축소해 학습 시간을 줄이고 성능을 향상시키는 방법을 알아본다.

### 1. 데이터 불러오기

```python
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/anonymous.csv'  
anonymous = pd.read_csv(file_url)
```

### 2. 데이터셋 확인

```python
print(anonymous.head())  
print(anonymous['class'].mean())    # 종속변수의 평균 확인  
print(anonymous.isna().sum().sum())     # 결측치 확인
```

- 종속변수의 평균은 `0.25` 이다.
- 독립변수가 총 4296개로 너무 많기 때문에 결측치 총 합을 출력했다.
	- 출력결과 결측치 총 합은 `0` 으로 결측치 처리 이슈는 없다.

### 3. PCA에 따른 모델링 성능/결과 비교

#### 3.1 데이터셋 분할

```python
from sklearn.model_selection import train_test_split  
  
X = anonymous.drop('class', axis=1)  
y = anonymous['class']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
```

#### 3.2 스케일링

PCA에서도 변수 간의 스케일을 일정하게 맞춰주는 것이 중요하다.

```python
from sklearn.preprocessing import StandardScaler  
  
scaler = StandardScaler()  
scaler.fit(X_train)  
  
X_train_scaled = scaler.transform(X_train)  
X_test_scaled = scaler.transform(X_test)
```

#### 3.3 모델링 및 평가

```python
from sklearn.ensemble import RandomForestClassifier  
import time

model_1 = RandomForestClassifier(random_state=100)
start_time = time.time()
model_1.fit(X_train_scaled, y_train)
print(time.time() - start_time)  # 소요 시간 측정
  
from sklearn.metrics import accuracy_score, roc_auc_score  
  
pred_1 = model_1.predict(X_test_scaled)  
print(accuracy_score(y_test, pred_1))  
proba_1 = model_1.predict_proba(X_test_scaled)  
print(roc_auc_score(y_test, proba_1[:, 1]))
```

- PCA 사용 전, 학습에 걸리는 시간을 측정한다.
	- 소요 시간은 `97초` 가 걸렸다.
- 정확도는 `약 96%` 로 좋은 예측 결과이다.
- AUC는 `0.99` 이상으로 굉장히 높은 값이 나왔다.

#### 3.4 PCA

우선 몇개의 주성분으로 만들 것인지 정해야 한다.
테스트 삼아 2개의 주성분으로 지정해 학습해본다.
- 4000개가 넘는 변수를 2개의 차원으로 축소하면 데이터 손실이 너무 클 것이 예상된다.

```python
pca = PCA(n_components=2)  
pca.fit(X_train_scaled)  
print(pca.explained_variance_ratio_)    # 데이터 반영 비율 확인
```

- 데이터 반영 비율을 확인했을때 기존 데이터의 `0.08` 정도의 정보만 반영한다.

최적의 주성분 개수는 주관적인 판단에 의한 것이다.
하지만, 엘보우 기법을 이용하면 조금 도움을 받을 수도 있다.

```python
var_ratio = []  
for i in range(100, 550, 50):  
    pca = PCA(n_components=i)  
    pca.fit_transform(X_train_scaled)  
    ratio = pca.explained_variance_ratio_.sum()  
    var_ratio.append(ratio)  
  
# 그래프로 확인  
sns.lineplot(x=range(100, 550, 50), y=var_ratio)  
plt.show()
```

 - `100~500` 사이의 주성분 개수에서 얻을 수 있는 데이터 반영 비율은 약 `62% ~ 82%` 정도이다.
 - 그래프가 드라마틱하게 꺾이는 부분이 없기 때문에 한눈에 적정값을 찾을 수는 없지만, 본인 기준에 맞는 적정값을 찾는 가이드라인으로 삼을 수 있다.

#### 3.5 주성분이 400개인 데이터 만들기

정보 반영 비율을 약 80%를 기준으로 잡고, 이에 근사치인 `400` 을 채택하여 주성분이 400개인 데이터를 만든다.

```python
pca = PCA(n_components=400, random_state=100)  
pca.fit(X_train_scaled)  
X_train_scaled_pca = pca.transform(X_train_scaled)  
X_test_scaled_pca = pca.transform(X_test_scaled)
```

#### 3.6 랜덤 포레스트 예측 모델

```python
import time

model_2 = RandomForestClassifier(random_state=100)
start_time = time.time()
model_2.fit(X_train_scaled_pca, y_train)
print(time.time() - start_time)  # 소요 시간 확인
  
pred_2 = model_2.predict(X_test_scaled_pca)  
print(accuracy_score(y_test, pred_2))  
proba_2 = model_2.predict_proba(X_test_scaled_pca)  
print(roc_auc_score(y_test, proba_2[:, 1]))
```

- 소요 시간을 측정한 결과 `62초` 로 학습 시간이 대폭 줄어들었다.
- 정확도는 `98%` 로 기존보다 약간 높은 정확도를 보여준다.
- AUC는 `0.99` 이상으로 기존과 거의 유사한 수준의 예측력을 보여준다.

PCA를 통해 학습 시간을 대폭 줄이고 거의 동일한 수준의 에측력을 보여주었다.
PCA가 언제나 이러한 성과를 내는 것은 아니다. 하지만 그것이 잘못된 것은 아니며 PCA를 사용하기 적합한 상황이 아닐 뿐이다.

## 🔥 이해하기 : PCA

PCA는 특성을 최대한 유지하는 방향으로 차원 축소를 진행한다.

### 1. 3차원을 2차워으로 축소하는 예시

3차원 공간에 모빌 2개가 천장에 매달려 있다. 만약 사진을 찍어 2개의 모빌 위치를 최대한 잘 담아내려면 어떻게 해야 할까? 사진을 찍는 위치에 따라 모빌 간의 거리가 다르게 보일 것이다.

사진 속 모빌 간의 거리가 데이터의 기존 분산을 얼마나 유지하는지를 의미한다. 모빌이 겹쳐보인다면 데이터 분산이 0이 되도록 차원이 축소된 것이다.

PCA는 데이터들의 분산을 최대한 담아내면서 차원을 축소한다.

### 2. 2차원을 1차원으로 축소하는 예시

2차원 평면 그래프를 1차원인 선으로 축소하는 경우를 다뤄본다.

먼저, 간단히 생각해서 X축과 Y축에 해당하는 선으로 차원 축소를 할 수 있다.
또한 2차원 데이터의 X에 대한 분포와 Y에 대한 분포를 최대한 손실이 없게끔 표현하는 대각선의 선 위로 차원 축소를 할 수도 있을 것이다.
PCA가 이런 다양한 각도에서 투영해 최적의 대각선을 찾아낸다.

PCA는 최적의 대각선을 찾은 후 그것에 대하여 직교하는 또 다른 선을 긋고 다시 한번 데이터를 투영시킨다.
이런 직교 선을 이용하는 이유는 첫번째 주성분이 담아내지 못한 특성을 최대한 담아낼 수 있는 방향이기 때문이다. 