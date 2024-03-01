# 04. 선형 회귀 : 보험료 예측하기

## 🔥 선형 회귀 (Linear Regression) 이란?

**선형 회귀 (Linear Regression)** 는 가장 기초적인 머신러닝 모델로, 여러가지 데이터를 활용해 연속형 변수인 목표 변수를 예측해 내는 것이 목적이다. 
복잡한 알고리즘에 비해 예측력이 떨어지지만 데이터의 특성이 복잡하지 않을 때 쉽고 빠른 예측이 가능하다. 또한, 다른 모델과의 성능을 비교하는 baseline 으로 사용하기도 한다.

- 장점
	- 모델이 간단해 구현과 해석이 쉽다.
	- 모델링하는 데 오랜 시간이 걸리지 않는다.
- 단점
	- 최신 알고리즘에 비해 예측력이 떨어진다.
	- 독립변수와 예측변수의 선형 관계를 전제로 하므로, 전제에서 벗어나는 데이터는 좋은 예측을 보여주기 어렵다.

## 🔥 보험 데이터셋을 이용한 보험료 예측하기

Linear Regression 알고리즘을 이용해 보험사에서 청구할 보험료를 예측하는 모델을 학습시키고, 성능을 평가한다. 데이터셋은 보험사에서 청구하는 병원 비용이 **종속변수** 이며 나이, 성별, BMI, 자녀 수, 흡연 여부를 **독립변수** 로 사용한다.

### 1. 라이브러리 및 데이터 불러오기
```python
import pandas as pd

# 데이터 불러오기
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/insurance.csv'
data = pd.read_csv(file_url)

# 데이터셋에 대한 다양한 정보 출력하기
print(data)
print(data.head())
print(data.info())
print(round(data.describe()), 2)
```

### 2. 전처리: Train Dataset, Test Dataset Split

데이터를 나누는 작업은 크게 2가지 차원으로 진행된다.
- 종속변수와 독립변수의 분리
- 훈련 데이터와 테스트 데이터의 분리
이렇게 총 4개의 데이터셋으로 분리한다.

데이터를 훈련용과 테스트용으로 나누는 이유는 당연하게도 모델의 성능을 높이기 위함이다. 훈련 데이터와 테스트 데이터로 같은 데이터를 사용한다면 새로운 데이터에 대한 예측력이 떨어지는 즉, 일반화 성능이 낮아지기 때문이다.

```python
# 종속변수와 독립변수의 분리
X = data[['age', 'sex', 'bmi', 'children', 'smoker']]
y = data['charges']

# 훈련 데이터와 테스트 데이터의 분리
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
```

**trian_test_split()** 함수의 option 들을 살펴보면 다음과 같다.
- test_size 는 훈련 데이터와 테스트 데이터를 나누는 비율을 의미한다.
	- 보통 7:3, 8:2 로 많이 나누며 validation 까지 진행할 경우 7:2:1 로 많이 나눈다.
- random_state 는 임의의 값을 선택하고, 다음 분할에서도 같은 값을 설정하면 랜덤하지만 같은 분포의 데이터로 나누어준다.

### 3. 모델링

모델링은 머신러닝 알고리즘으로 모델을 학습시키는 과정이며, 결과물은 머신러닝 모델이 된다. 사용할 머신러닝 알고리즘을 선택하고 독립변수와 종속변수를 **fit()** 함수에 인자로 주어 학습한다.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### 4. 모델을 활용해 예측하기

다음과 같이 학습된 모델의 **predict()** 함수에 테스트 데이터를 인자로 넣으면 예측 결과물이 **pred** 변수에 담긴다.
```python
pred = model.predict(X_test)
```

### 5. 모델 평가하기

모델을 평가하는 방법으로 크게 3가지 방법이 있다.
- 테이블로 평가
- 그래프로 평가
- 통계적인 방법으로 평가

```python
# 테이블로 평가하기
comparison = pd.DataFrame({'actual': y_test, 'pred': pred})
print(comparison)

# 그래프로 평가하기 (산점도 그래프)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,10))
sns.scatterplot(x='actual', y='pred', data=comparison)

# 통계적인 방법으로 평가하기
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, pred) ** 0.5)

```
먼저, 테이블로 평가하는 방법은 실제값과 예측값을 직접 비교하는 것이다. 두 값의 차이를 통해 모델이 얼마나 예측을 잘하고 있는지 판단할 수 있다.

두번째는, 그래프를 통해 판단하는 것이다. 테이블의 모든 데이터를 직접 비교하는 것은 어렵기 때문에 그래프를 이용한다면 보다 더 쉽게 모델을 평가할 수 있다. 산점도 그래프를 그릴 경우 예측값과 실제값이 1:1 로 매칭되는 직선을 기준으로 평가를 하면 된다. 여기에 가까울 수록 예측을 잘했다고 평가할 수 있다.

세번째는, 통계적인 방법을 이용하는 것이다. 연속형 변수를 예측하고 평가할 때 가장 흔하게 쓰이는 **RMSE (Root Mean Squared Error)** 를 이용할 수 있다. 이것은 실제값과 예측값 사이의 오차를 각각 합산하는 개념이다. 
- MAE : 평균 절대 오차, 실제값과 예측값 사이의 오차에 절대값을 씌운 후 이에 대한 평균을 계산한다. 값이 작을 수록 좋은 지표이다.
- MSE : 평균 제곱 오차, 실제값과 예측값 사이의 오차를 제곱하고 이에 대한 평균을 계산한다. 값이 작을 수록 좋은 지표이다.
- RMSE : 루트 평균 제곱 오차, MSE에 루트를 씌운 값으로 가장 일반적으로 사용된다.

## 🔥 마무리

**Linear Regression** 은 독립변수와 종속변수 간에 선형 관계가 있음을 가정하여 최적의 선을 그려서 예측하는 방법이다. 최적의 선은 머신러닝에서 **손실 함수** 를 최소화하는 선을 찾아 모델을 만들어낸다. 손실 함수는 예측값과 실제값의 차이 (오차) 를 평가하는 방법을 말한다. 

Linear Regresison 문제에 활용할 수 있는 관련 모델은 다음과 같다.
- Ridge Regression
	- 선형 회귀 모델에 L2 정규화를 적용한 모델로 Overfitting 을 억제하는 효과
- Lasso Regression
	- 선형 회귀 모델에 L1 정규화를 적용한 모델로 Feature Selection 및 Overfitting 을 억제하는 효과
- Elastic Regression
	- Ridge 와 Lasso 의 단점을 절충시킨 모델