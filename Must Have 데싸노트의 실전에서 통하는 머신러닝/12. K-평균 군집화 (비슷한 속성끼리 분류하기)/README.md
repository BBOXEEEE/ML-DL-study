# 12. K-평균 군집화 : 비슷한 속성끼리 분류하기

## 🔥 K-means Clustering 이란?

**K-means Clustering** 이란, 비지도 학습의 대표적 알고리즘으로 목표 변수가 없는 상태에서 데이터를 비슷한 유형끼리 묶어내는 머신러닝 기법이다.
거리 기반으로 작동하며 적절한 K값을 사용자가 지정해야 한다.
거리 기반으로 작동하기 때문에 데이터 위치가 가까운 데이터끼리 한 그룹으로 묶으며, 전체 그룹의 수는 사용자가 지정한 K개이다.

- 장점
	- 구현이 비교적 간단하다.
	- 클러스터링 결과를 쉽게 해석할 수 있다.
- 단점
	- 최적의 K값을 자동으로 찾지 못하고, 사용자가 직접 선택해야 한다.
	- 거리 기반 알고리즘이기 때문에 변수의 스케일에 따라 다른 결과를 나타낼 수 있다.

## 🔥 데이터를 비슷한 속성끼리 분류하기

### 1. 인위적으로 만든 데이터셋

**K-means Clustering** 을 학습할 목적으로 만들어진 인위적인 데이터로 K-means Clustering 에 대해 알아본다. 인위적인 데이터로, 변수들에는 아무런 의미가 없다.

### 1.1 데이터셋 불러오기

```python
import pandas as pd

file_url = 'https://raw.githubusercontent.com/musthave-ML10/data_source/main/example_cluster.csv'  
data = pd.read_csv(file_url)
```

### 1.2 데이터셋 확인

```python
import matplotlib.pyplot as plt  
import seaborn as sns

print(data)
# 산점도
sns.scatterplot(x='var_1', y='var_2', data=data)
plt.show()
```

- 데이터셋에는 총 1000개의 데이터와 2개의 변수가 있다.
- 산점도로 확인했을 때 한눈에 보기에도 크게 3가지 그룹으로 나뉘어 있다.

### 1.3 연습용 데이터 모델링 및 평가

연습용 데이터를 이용해 데이터를 3개의 그룹으로 나눈다.

```python
from sklearn.cluster import KMeans  
  
kmeans_model = KMeans(n_clusters=3, random_state=100)  
kmeans_model.fit(data)  
data['label'] = kmeans_model.predict(data)  
  
# 산점도 확인  
sns.scatterplot(x='var_1', y='var_2', data=data, hue='label', palette='rainbow')
plt.show()
```

- 모델 예측 결과를 산점도로 확인했을 때, 데이터셋을 확인했던 것과 같이 3개의 그룹으로 나뉘어진 것을 확인할 수 있다.

### 1.4 엘보우 기법으로 최적의 K값 구하기

실제 데이터셋은 너무 많은 변수들이 있어 그래프로 확인하기도 어려우며 사람의 눈으로 보기에 정확히 몇 개의 그룹인지 구분이 애매한 경우가 많다.
이러한 경우 K값을 지정하기 쉽지 않은데, 이러한 경우 **엘보우 기법 (ellbow method)** 을 활용할 수 있다.

- 각 그룹에서 중심과 각 그룹에 해당하는 데이터 간의 거리에 대한 합을 계산한다.
- 이 값을 **이너셔 (inertia)** 혹은 **관성** 이라고 한다.

**inertia** 는 클러스터의 중점과 데이터 간의 거리이기 때문에 작을수록 그룹별로 더 오밀조밀 잘 모이게 분류했다고 할 수 있다.
하지만, K값이 커지면 이는 필연적으로 작아지게 된다.
따라서 클러스터 수를 가급적 적게 유지하면서 동시에 거리의 합이 어느정도 작은 적절한 K값이 필요하다.

```python
distance = []  
for k in range(2, 10):  
    k_model = KMeans(n_clusters=k)  
    k_model.fit(data)  
    distance.append(k_model.inertia_)  
  
print(distance)  
sns.lineplot(x=range(2, 10), y=distance)  
plt.show()
```

- `distance` 값을 산점도로 확인했을 때 K가 3인 지점에서 거리의 합이 크게 감소한다.
- 그 후로는 완만하게 감소한다.
- 그래프를 확인했을 때 사람의 팔꿈치 모양과 비슷하다고 해서 엘보우 기법이라 부른다.
- 엘보우 기법은 이와 같이 거리의 합이 급격히 줄어드는 K값을 포착해 최적의 K값을 찾도록 도와주는 방법이다.

### 2. 고객 데이터셋 활용

고객 데이터셋을 이용해 K-means Clustering 을 해본다.

### 2.1 데이터셋 불러오기

```python
file_url = 'https://raw.githubusercontent.com/musthave-ML10/data_source/main/customer.csv'  
customer = pd.read_csv(file_url)
```

### 2.2 데이터셋 확인

```python
print(data.head())  
print(customer['cc_num'].nunique())     # 고유값 확인 : 고객의 수  
print(customer['category'].nunique())   # 고유값 확인 : 범주의 수
```

- 총 100명의 고객이 있고, 범주는 11개가 있다.

### 2.3 전처리 : 피처 엔지니어링

범주별 금액을 계산하기 위해 `category` 변수를 더미 변수로 변환하고, 범주별 얼만큼의 금액을 썼는지 계산한다. 그 후에 고객별 총 사용 금액 및 범주별 사용 금액을 구한다. 또한, 데이터 스케일의 영향을 받기 때문에 데이터를 스케일링 해준다.

```python
# (1) 더미 변수 변환  
customer_dummy = pd.get_dummies(customer, columns=['category'])  
cat_list = customer_dummy.columns[2:]   # 변수 이름 리스트 생성  
  
# 금액으로 변수 업데이트  
for i in cat_list:  
    customer_dummy[i] = customer_dummy[i] * customer_dummy['amt']  
  
# cc_num 별 총 사용 금액  
customer_agg = customer_dummy.groupby('cc_num').sum()  
  
# (2) 스케일링  
from sklearn.preprocessing import StandardScaler  
  
scaler = StandardScaler()  
scaled_df = pd.DataFrame(scaler.fit_transform(customer_agg),  
                         columns=customer_agg.columns,  
                         index=customer_agg.index)
```

### 2.4 모델링 및 실루엣 계수

#### 2.4.1 엘보우 기법으로 최적의 K값 찾기

K값을 전혀 예상할 수 없기 때문에 엘보우 기법을 사용해 K값을 찾아본다.

```python
distance = []  
for k in range(2, 10):  
    k_model = KMeans(n_clusters=k)  
    k_model.fit(scaled_df)  
    labels = k_model.predict(scaled_df)  
    distance.append(k_model.inertia_)  
  
sns.lineplot(x=range(2, 10), y=distance)  
plt.show()
```

- 산점도 그래프에서 이전과 같이 거리의 합이 급격하게 감소하는 지점이 없다.
- 이러한 경우 K값을 결정하기 상당히 어렵다.

#### 2.4.2 실루엣 계수

엘보우 기법으로 최적의 K값을 찾을 수 없는 경우 **실루엣 계수 (silhouette coefficient)** 를 사용할 수 있다.

- 클러스터 내부에서의 평균 거리와 최근접한 다른 클러스터 데이터와의 평균 거리도 점수에 반영한다.

```python
from sklearn.metrics import silhouette_score  
  
silhouette = []  
for k in range(2, 10):  
    k_model = KMeans(n_clusters=k)  
    k_model.fit(scaled_df)  
    labels = k_model.predict(scaled_df)  
    silhouette.append(silhouette_score(scaled_df, labels))  
  
sns.lineplot(x=range(2, 10), y=silhouette)  
plt.show()
```

- 실루엣 계수는 높은 값일수록 더 좋은 분류를 의미한다.
- 그래프를 통해 K는 4일 때 가장 좋은 분류 성능을 보이는 것을 확인할 수 있다.

### 5. 최종 모델링 및 결과 해석

```python
# (1) 모델링  
k_model = KMeans(n_clusters=4)  
k_model.fit(scaled_df)  
labels = k_model.predict(scaled_df)  
scaled_df['label'] = labels  
  
# (2) label 별 데이터 요약  
# label 별 평균값
scaled_df_mean = scaled_df.groupby('label').mean()  
# label 별 등장 횟수
scaled_df_count = scaled_df.groupby('label').count()['category_travel']
# 변수 이름 수정 
scaled_df_count = scaled_df_count.rename('count') 
# 데이터 합치기 
scaled_df_all = scaled_df_mean.join(scaled_df_count)

print(scaled_df_all)
```

