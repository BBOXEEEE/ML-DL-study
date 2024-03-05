# 06. K-최근접 이웃 (KNN) : 와인 등급 예측하기

## 🔥 K-최근접 이웃이란?

**K-최근접 이웃 (K Nearest  Neighbors)** 이란, 거리 기반 모델로 선형 관계를 전제로 하지 않는다.
각 데이터 간의 거리를 활용해 새로운 데이터를 예측하는 모델이다.
이때, 가까이 있는 데이터를 고려하여 예측값이 결정된다.

- 장점
	- 수식에 대한 설명이 필요 없을 만큼 직관적이고 간단하다.
	- 선형 모델과 다르게 별도의 가정이 없다. (ex. feature 와 target 의 선형 관계)
- 단점
	- 데이터가 커질수록 상당히 느려질 수 있다.
	- 아웃라이어 (평균치에서 크게 벗어나는 데이터, 이상치) 에 취약하다.

## 🔥 와인 데이터셋을 이용해 와인 등급 예측하기

와인에 대한 데이터를 기반으로 3가지 목표값 (와인 등급)을 예측한다. 3가지 목표값으로 이루어진 범주형 변수이므로 다중분류 문제에 해당한다.

### 1. 라이브러리 및 데이터 불러오기

```python
import pandas as pd

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/wine.csv'
data = pd.read_csv(file_url)
```

### 2. 데이터셋 확인

데이터셋을 살펴보면 다음과 같다.
- Feature
	- alcohol :알코올(도수)
	- malic_acid : 알산(사과산)
	- ash : 증발/소각 후 남은 무기불
	- alcalinity_of_ash : 남은 무기물의 알칼리성
	- magnesium : 마그네슘
	- total_phenols : 전체 페놀
	- flavanoids : 플라보노이드(색소)
	- nonflavanoid_phenols : 비색소 페놀
	- proanthocyanins : 프로안토시아닌
	- color_intensity : 색상 강도
	- hue : 색조
	- od280/od315_of_diluted_wines : 희석된 와인의 단백질 함량
	- proline : 프롤린
- Target
	- class : 와인 등급

```python
print(data.head())
print(data.info())
print(data.describe())
```

### 3. Target 에서 고유값 확인

Target 의 특성에 따라 즉, 연속형 변수인지, 이진변수인지, 3개 이상으로 된 범주형 변수인지에 따라 적합한 알고리즘이 다르기 때문에 필요에 따라 target 의 고유값을 확인한다.

```python
print(data['class'].unique()) # 고유값 출력
print(data['class'].nunique()) # 고유값 개수 출력
print(data['class'].value_counts()) # 고유값에 해당하는 개수 출력

sns.barplot(x=data['class'].value_counts().index, y=data['class'].value_counts())
plt.show() # 막대 그래프로 출력
```

### 4. 전처리 : 결측치 처리

`data.info()` 를 통해 데이터에 결측치가 있는 것을 확인할 수 있다.
그 외에도 `isna()` 메소드를 통해 결측치를 확인하는 방법이 있다.

```python
print(data.isna()) # 결측치 여부 출력
print(data.isna().sum()) # isna() 와 sum() 을 조합해 결측치 개수 출력
print(data.isna().mean()) # 결측치 비율 출력
```

교재에서 제시하는 결측치 제거 방법은 크게 3가지가 있다.
- 결측치 행 제거 : **dropna()**
	- 결측치가 있는 행을 지우는 것이다.
- 결측 변수 제거 : **drop()**
	- 결측치가 있는 변수 자체를 지우는 것이다.
- 결측값 채우기 : **fillna()**
	- 결측치에 값을 채워넣는 방법으로 일반적으로 평균값이나 median 을 이용한다.

```python
# 결측치 행 제거
data.dropna()
data.dropna(subset=['alcohol'])

# 결측 변수 제거
data.drop['alcohol', 'nonflavanoid_phenols', axis=1]

# 결측값 채우기
data.fillna(data.mean())
data.fillna(data.median())
```

이제 결측치를 처리하는 방식을 선택한다.

일반적인 방법은  **dropna()** 를 사용하여 결측치 행을 지우는 것이다. 평균 등을 이용해 결측치를 채우면 아무리 비슷한 값을 채우더라도 실제값과 일치할 가능성이 매우 낮기 때문에 오차의 원인이 될 수 있다. 즉, 데이터에 **노이즈** 가 더해진 효과를 내게 된다.

하지만, 결측치 행을 지우는 방법은 경우에 따라 너무 많은 데이터가 삭제될 수 있다는 단점이 있다. 만약 특정 변수의 90%가 결측치라면 90% 데이터가 삭제된다. 따라서, 행을 지우는 방식을 선택하려면 결측치 비중이 매우 낮아야하고, 데이터 크기도 충분히 커야한다.
현재 데이터셋은 178개의 데이터 밖에 없기 때문에 행 삭제 방식을 사용하지 않는다.

변수 자체를 **drop()** 하는 방식은 매우 무모한 방식일 수 있다. 통상적으로 해당 변수의 결측치가 50% 이상이면 drop() 을 고려해볼 만하고, 70~80% 이상이면 가급적 drop() 을 적용하는 것이 좋다. 하지만, 경우에 따라서 해당 변수가 프로젝트에 매우 중요한 역할을 한다면 어떻게든 활용 방법을 찾는 것이 좋다.

여기서 사용하는 **결측치 처리 방법** 은 **fillna()** 로 결측치를 채우되, 아웃라이어에 조금 덜 민감한 중윗값을 사용한다.

```python
data.fillna(data.median(), inplace=True)
print(data.isna().mean()) # 결측치 처리 확인
```

### 5. 스케일링

현재 데이터셋 값들의 범위가 다양하다. K-최근접 이웃은 거리 기반의 알고리즘이기 때문에 스케일 문제가 안 좋은 결과를 초래할 수 있다.
- **alcohol (min: 11.03, max: 14.75)**
- **magnesium (min: 70, max: 162)**
- 위 두 변수에서 1이라는 값이 차지하는 의미가 상당히 다르다.

스케일링은 4가지 방법이 있다.
- 표준화 스케일링 : 평균이 0이 되고, 표준편차가 1이 되도록 데이터를 고르게 분포시키는 데 사용
- 로버스트 스케일링 : 데이터에 아웃라이어가 존재하고, 그 영향력을 그대로 유지하고 싶을 때 사용
- 최소-최대 스케일링 : 데이터 분포의 특성을 최대한 그대로 유지하고 싶을 때 사용
- 정규화 스케일링 : 행 기준의 스케일링이 필요할 때 사용하나, 실제로 거의 사용하지 않음.

#### 5.1 표준화 스케일링

```python
from sklearn.preprocessing import StandardScaler

st_scaler = StandardScaler()
st_scaler.fit(data)
st_scaled = st_scaler.transform(data)
st_scaled = pd.DataFrame(st_scaled, columns=data.columns)
```

- `fit()` 메소드로 가진 데이터를 학습시킨다. 이 단계에서 스케일링에 필요한 정보 (평균, 표준편차)가 학습된다.
- `transform()` 메소드로 연산을 하는데, 학습에서 얻은 정보로 계산하게 된다.

표준화 스케일은 데이터를 표준화된 **정규분포** 로 만들어주는 방법이며, 아래와 같은 공식으로 계산된다. 
- **(변수의 i번째 값 - 해당 변수의 평균) / 해당 변수의 표준편차** 

#### 5.2 로버스트 스케일링

```python
from sklearn.preprocessing import RobustScaler

rb_scaler = RobustScaler()
rb_scaled = rb_scaler.fit_transform(data)
rb_scaled = pd.DataFrame(rb_scaled, columns=data.columns)
```

표준화 스케일링과 달리 로버스트 스케일링은 사분위 값을 이용하며, 아래와 같은 공식으로 계산된다.
- **(변수의 i번째 값 - 변수의 중위값) / (75% 지점 - 25% 지점)

#### 5.3 최소-최대 스케일링

```python
from sklearn.preprocessing import MinMaxScaler

mm_scaler = MinMaxScaler()
mm_scaled = mm_scaler.fit_transform(data)
mm_scaled = pd.DataFrame(mm_scaled, columns=data.columns)
```

최소-최대 스케일링은 모든 컬럼에서 최대값이 1, 최소값이 0인 형태로 변환된다는 특징이 있다.
아래와 같은 공식으로 계산된다.
- **(변수의 i번째 값 - 변수의 최소값) / (최대값 - 최소값)**

#### 5.4 스케일링 방법 선택

우선, 아웃라이너 유무에 따라 스케일링 방법을 선택한다.
- 아웃라이어의 영향이 큰 데이터이고, 이를 피하고 싶다면 **로버스트 스케일링** 이 적합하다.
- 로버스트 스케일링은 사분위값을 사용하므로 아웃라이어에 영향을 덜 받는다.

때에 따라 데이터의 기존 분포를 최대한 유지하여 스케일링 하는 것이 필요할 수 있다.
- 이러한 경우 **최소-최대 스케일링** 이 적합하다.
- 최소-최대 스케일링은 최대값 1과 최소값 0의 범위에서 기존 데이터의 분포를 최대한 그대로 옮겨담아 낸다.

**표준화 스케일링** 은 기존 데이터가 정규분포를 따르고 아웃라이너가 없는 상황에서 무난하게 사용된다.

**스케일링 적용 시 주의할 점** 은 다음과 같다.
- 스케일링 대상에서 종속변수를 제외한다.
- 스케일링 전 훈련셋과 테스트셋을 나누어야 한다.

```python
from sklearn.model_selection import train_test_split

# 데이터 나누기
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# 스케일링 적용 : 최소-최대 스케일링
mm_scaler = MinMaxScaler()
mm_scaler.fit(X_train)
X_train_scaled = mm_scaler.transform(X_train)
X_test scaled = mm_scaler.transform(X_test)
```

### 6. 모델링 및 예측/평가

```python
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score  
  
knn = KNeighborsClassifier()  
knn.fit(X_train_scaled, y_train)  
pred = knn.predict(X_test_scaled)  
print(accuracy_score(y_test, pred))
```

예측 결과 정확도는 **0.8888888888888888** 로, 약 **89%** 의 정확도를 보여준다.

### 7. 하이퍼파라미터 튜닝하기

KNN 알고리즘에는 **n_neighbors** 라는 중요한 파라미터가 있다. 예측에 가까운 이웃을 몇 개나 고려할지 정하는 파라미터이다. 기본값은 5이다.

값을 1부터 20까지 반복하면 어떤 파라미터 값이 높은 정확도를 보이는지 확인한다.

```python
scores = []

for i in range(1, 21):
	knn = KNeighborsClassifier(n_neighbors=i)
	knn.fit(X_train_scaled, y_train)  
	pred = knn.predict(X_test_scaled)  
	scores.append(accuracy_score(y_test, pred))

print(scores)
```

결과적으로 13 이후로는 더 나은 개선이 보이지 않는다. 이 값이 높을수록 연산량이 많아지기 때문에 13으로 선택하면 정확도 **97.2%** 가 나오는 것을 알 수 있다.

