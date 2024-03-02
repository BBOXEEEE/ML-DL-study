# 05. 로지스틱 회귀 : 타이타닉 생존자 예측하기

## 🔥 로지스틱 회귀 (Logistic Regression) 이란?

**로지스틱 회귀 (Logistic Regression)** 역시 Linear Regression 처럼 기본 분석 모델이다. Linear Regression 과 유사하지만 다루는 문제가 다르다. 연속된 변수를 예측하는 것이 아닌, **Yes/No** 와 같은 이진 분류 문제를 다룬다.

- 장점
	- 선형 회귀 분석만큼 구현하기 용이하다.
	- 계수를 사용해 각 변수의 중요성을 쉽게 파악할 수 있다.
- 단점
	- 선형 관계가 아닌 데이터에 대한 예측력이 떨어진다.

## 🔥 타이타닉 데이터셋을 이용한 생존자 예측하기

타이타닉 데이터셋에 포함된 이름, 나이, 성별, 티켓 등급 등을 활용해 생존 여부를 예측한다. **Survived (생존여부)** 가 종속변수이며 이 외의 변수들이 독립변수 (Feature) 이다.

### 1. 라이브러리 및 데이터 불러오기
```python
import pandas as pd

# 데이터 불러오기
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/titanic.csv'
data = pd.read_csv(file_url)

# 데이터셋 정보 출력
print(data.head())
print(data.info())
print(data.describe())
print(data.corr())     # 데이터 간 상관관계 확인
```

### 2. 데이터셋 확인

데이터셋을 살펴보면 다음과 같다.
- Feature
	- PClass : 티켓 클래스
	- Name : 이름
	- Sex : 성별
	- Age : 나이
	- SibSp : 함께 탑승한 형제 및 배우자 수
	- Parse : 함께 탑승한 부모 및 자녀의 수
	- Ticket : 티켓 번호
	- Embarked : 승선 항구 (C, Q, S)
- Target
	- Survived : 생존 여부

데이터 간의 상관관계를 파악하려면 다음과 같은 방법을 사용할 수 있다.
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(data.corr()) # 히트맵 생성
plt.show()

sns.heatmap(data.corr(), cmap='coolwarm') # 보기 좋은 컬러로 변경
plt.show()

sns.heatmap(data.corr(), cmap='coolwarm', vmin=-1, vmax=1) # 범위조정
plt.show()

sns.heatmap(data.corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=True) # 수치 출력
plt.show()
```

### 3. 데이터 전처리 : 범주형 변수 변환 (one-hot encoding)

타이타닉 데이터셋에는 자료형이 **object** 인 변수들이 4개 있다.
기본적으로 머신러닝 알고리즘에서 문자열로 된 데이터를 이해하지 못하기 때문에 이를 수치화하는 작업이 필요하다. 이때, 계절을 나타내는 범주형 변수를 단순히 1~4 로 표현한다면 숫자를 상대적 서열로 인식하기 때문에 별도의 컬럼을 추가하는 더미변수 즉, **one-hot encoding** 을 해야 한다.

```python
print(data['Name'].nunique())  # 889
print(data['Sex'].nunique())  # 2
print(data['Ticket'].nunique())  # 680
print(data['Embarked'].nunique())  # 3
```
**nunique()** 메소드를 사용하면 변수의 고유값 개수를 확인할 수 있다. 성별과 탑승 항구를 제외하면 고유값이 매우 많은 것을 알 수 있다. 이것을 모두 one-hot encoding 을 진행할 경우 컬럼이 엄청나게 늘어나 이런 경우 데이터를 제거하는 것을 고려할 수 있다. 이름과 티켓 번호는 학습에 유의미한 결과를 미치는 feature 가 아니기 때문에 데이터셋에서 제거하고, 성별과 탑승 항구에 대해서 one-hot encoding 을 진행한다.

```python
# 필요없는 데이터 제거
data = data.drop(['Name', 'Ticket'], axis=1)

# one-hot encoding
data = pd.get_dumies(data, columns=['Sex', 'Embarked'], drop_first=True)

# 데이터셋 확인
print(data.head())
```

one-hot encoding 을 할 때, **drop_first** 옵션을 True 로 설정하면 컬럼개수를 1개 줄일 수 있다. 성별의 경우 male 의 값이 1이면 남성, 0이면 여성임을 알 수 있기 때문에 불필요한 컬럼 1개를 줄이는 것이 계산량을 줄여줄 수 있다.

### 4. 모델링 및 평가

4장에서와 같은 방법으로 모델을 학습시키고 이것을 평가한다.

```python
from sklearn.model_selection import train_test_split

# data split
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

from sklearn.linear_model import LogisticRegression

# 모델 학습 및 예측
model = LogisticRegressioin()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

예측값과 실제값을 이용해 모델을 평가한다. 4장에서와 같은 방법인 RMSE 는 이진분류 문제 평가에 적합하지 않다. 다양한 이진분류 평가 지표로 `정확도, 오차행렬, 정밀도, 재현율, F1 Score, 민감도, 특이도, AUC` 등이 있는데 **정확도** 를 이용해 간단히 평가한다.

```python
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))
```
위 코드를 실행하면 정확도가 **0.7808988764044944** 로, 약 **78.08%** 가 나온다. 일반적으로 고유값 비율이 비슷한 비율로 분포되어 있다면, 80% 이상의 정확도는 나쁘지 않은 결과, 90% 이상이 괜찮은 결과로 보는 편이라고 한다. 

```python
print(model_coef_)
print(pd.Series(model.coef_[0], index=X.columns))
```
회귀 분석 모델의 계수를 확인한 결과는 다음과 같다.
- Pclass 는 음의 계수를 가지고 있기 때문에 Pclass 가 높을수록 생존 가능성이 낮다. Pclass는 낮은 숫자일수록 비행기의 퍼스트 클래스처럼 더 비싼 티켓이다.
- Age 는 낮을수록, 성별은 여성이 생존 가능성이 높다.

### 5. Feature Engineering

**Feature Engineering** 이란, 기존 데이터를 손보아 더 나은 변수를 만드는 기법이다. one-hot encoding 도 feature engineering 의 한 종류라고 할 수 있다. 
머신러닝에서 적합한 알고리즘을 선택하고 하이퍼파라미터를 튜닝하는 일도 중요하지만, 좋은 feature 를 하나 더 마련하는 일만큼 강력한 무기는 없다.

선형 모델에서는 **다중공선성 문제** 를 주의해야 한다. 이는 feature 사이에 상관관계가 높을 때 발생하는 문제이다. 예를 들어 독립변수 A와 B는 모두 목표변수를 양의 방향으로 이끄는 계수를 가지고 있을 때 상관관계가 매우 높다면, y가 증가한 이유가 A 때문인지 B 때문인지 명확하지 않다. 
다중공선성 문제는 **상관관계가 높은 변수 중 하나를 제거** 하거나 **둘을 모두 포괄시키는 새로운 변수** 를 만들거나, **PCA** 같은 방법으로 차원 축소를 수행해 해결할 수 있다.

타이타닉 데이터셋에서 Parch (함께 탑승한 부모 및 자녀의 수), SibSp (함께 탑승한 형제 및 배우자 수) 가 그나마 강한 상관관계를 보였기 때문에 이것을 포괄시키는 새로운 변수를 만든다.
- 부모와 자식, 형제와 자매는 모두 가족 구성원이라는 공통점이 있어 `Family` 라는 새로운 Feature 를 만든다.
```python
data['Family'] = data['SibSp'] + data['Parch']
data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
print(data.head())
```

이제 처음부터 모델을 재학습시키고 결과를 비교해본다.
```python
X = data.drop('Survived', axis=1)  
y = data['Survived']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)  
  
model = LogisticRegression()  
model.fit(X_train, y_train)  
  
pred = model.predict(X_test)  
print(accuracy_score(y_test, pred))
```
결과적으로 정확도는 **0.7921348314606742** 로 약 **79.21%** 로 약간의 정확도가 상승한 것을 확인할 수 있다.

**Feature Engineering** 에는 정답이 없다. 데이터의 특성이나 도메인 지식에 따라 무궁무진하게 확장할 수 있다. 혹은 그 어떤 Feature Engineering 도 수행할 수 없는 경우도 있어 매우 중요하면서도 아주 까다롭다.