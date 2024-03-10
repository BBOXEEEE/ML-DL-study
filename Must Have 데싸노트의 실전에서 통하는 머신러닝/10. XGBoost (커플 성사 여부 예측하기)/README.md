# 10. XGBoost : 커플 성사 여부 예측하기

## 🔥 XGBoost 란?

**Random Forest** 는 각 트리를 독립적으로 만드는 알고리즘이다.

반면, **Boosting** 은 순차적으로 트리를 만들어 이전 트리로부터 더 나은 트리를 만들어내는 알고리즘이다. 부스팅 알고리즘은 트리 모델을 기반으로 한 최신 알고리즘 중 하나로, 랜덤 포레스트보다 훨씬 더 빠른 속도와 좋은 예측 능력을 보여준다.
대표적인 알고리즘으로 **XGBoost, LightGBM, CatBoost** 등이 있다.

- 장점
	- 예측 속도가 빠르고 예측력 또한 좋다.
	- 변수 종류가 많고 데이터가 클수록 상대적으로 뛰어난 성능을 보인다.
- 단점
	- 복잡한 모델인 만큼, 해석에 어려움이 있다.
	- 더 나은 성능을 위한 하이퍼파라미터 튜닝이 까다롭다.

## 🔥 스피드데이팅 데이터셋을 이용한 커플 성사 여부 예측하기

### 1. 데이터셋 불러오기

```python
import pandas as pd

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/dating.csv'  
data = pd.read_csv(file_url)
```

### 2. 데이터셋 확인

```python
print(data.head())  
print(data.info())  
print(round(data.describe(), 2))
```

총 39개의 Feature 가 있다. 여기서 Target 은 `match` 이다.

피처 엔지니어링에 사용할 주요 Feature 는 다음과 같다.
- `has_null` : Null 값이 있는지 여부
- `age / age_o` : age는 본인 나이, age_o 는 상대방 나이
- `race / race_o` : 본인 인종과 상대방 인종
- `importance_same_race / importance_same_religion` : 인종과 종교를 중요시 여기는지에 대한 응답
- `attractive, sincere, intelligence, funny, ambitious, shared_interests` : 4가지 관점에서 평가되어 총 24개의 변수가 있다.
	- `pref_o_xxx` : 상대방이 xxx 항목을 얼마나 중요하게 생각하는지에 대한 응답
	- `xxx_o` : 상대방이 본인에 대한 xxx 항목을 평가한 항목
	- `xxx_important` : xxx 항목에 대해 본인이 얼마나 중요하게 생각하는가에 대한 응답
- `xxx_partner` : 본인이 상대바엥 대한 xxx 항목을 평가한 항목
- `intertests_correlate` : 관심사 연관도
- `expected_happy_with_sd_people` : 함께할 때 얼마나 좋을지에 대한 기대치
- `expected_num_interested_in_me` : 얼마나 많은 사람이 나에게 관심을 보일지에 대한 기대치
- `like` : 파트너가 마음에 들었는지 여부
- `guess_prob_liked` : 파트너가 나를 마음에 들어했을지에 대한 예상
- `met` : 파트너를 이전에 만난 적이 있는지 여부

### 3. 전처리 : 결측치 처리

**data.isna().mean()** 을 통해 결측치 비율을 확인한 결과 대부분 변수에서 결측치가 보이나 대체로 5% 미만이다.
-99 와 같이 응답하지 않음을 의미하는 숫자로 채워넣는다.
하지만, 중요도와 관련된 변수들은 **중요도 x 점수** 로 피처 엔지니어링을 하기 위해 결측치를 제거한다.

```python
# 중요도 관련 변수는 결측치 제거  
data = data.dropna(subset=['pref_o_attractive', 'pref_o_sincere', 
						   'pref_o_intelligence', 'pref_o_funny',  
                           'pref_o_ambitious', 'pref_o_shared_interests',
                           'attractive_important', 'sincere_important',
                           'intellicence_important', 'funny_important',  
                           'ambtition_important', 'shared_interests_important'])  
# 그 외의 결측치는 -99 (응답하지 않음) 으로 채움  
data = data.fillna(-99)
```

### 4. 전처리 : 피처 엔지니어링

#### 4.1 age

나이차가 얼마나 많이 나는지를 계산한다. 단순한 나이 차이보다 성별도 고려한다.

```python
def age_gap(x):  
    if x['age'] == -99:  
        return -99  
    elif x['age_o'] == -99:  
        return -99  
    elif x['gender'] == 'female':  
        return x['age_o'] - x['age']  
    else:  
        return x['age'] - x['age_o']  
  
  
data['age_gap'] = data.apply(age_gap, axis=1)   # age_gap 변수에 age_gap 함수 적용  
data['age_gap_abs'] = abs(data['age_gap'])  # 절대값 적용
```

- 결측치에 대한 처리를 해준다.
- 나이 차이 자체가 중요한 변수가 될 수도 있기 때문에 절대값을 적용해 새로운 변수를 추가한다.

#### 4.2 race

인종이 같으면 1, 다르면 -1로 처리한다.

```python
def same_race(x):  
    if x['race'] == -99:  
        return -99  
    elif x['race_o'] == -99:  
        return -99  
    elif x['race'] == x['race_o']:  
        return 1  
    else:  
        return -1  
  
  
data['same_race'] = data.apply(same_race, axis=1)   # same_race 함수 적용
```

또한, 동일 인종 여부가 얼마나 중요한지를 의미하는 변수에 대해 동일 인종을 구한 변수와 이 중요도를 곱해서 처리한다.

```python
def same_race_point(x):  
    if x['same_race'] == -99:  
        return -99  
    else:  
        return x['same_race'] * x['importance_same_race']  
  
  
data['same_race_point'] = data.apply(same_race_point, axis=1)
```

#### 4.3 평가와 관련된 변수

간단하게 평가점수 x 중요도로 계산하여 새로운 변수를 만든다.

```python
def rating(x, importance, score):  
    if x[importance] == -99:  
        return -99  
    elif x[score] == -99:  
        return -99  
    else:  
        return x[importance] * x[score]
```

다음과 같은 과정을 거쳐 이 함수를 적용한다.

```python
partner_imp = data.columns[8:14]    # 상대방의 중요도  
partner_rate_me = data.columns[14:20]   # 본인에 대한 상대방의 평가  
my_imp = data.columns[20:26]    # 본인의 중요도  
my_rate_partner = data.columns[26:32]   # 상대방에 대한 본인의 평가  
  
new_label_partner = ['attractive_p', 'sincere_partner_p', 'intelligence_p',  
                     'funny_p', 'ambition_p', 'shared_interests_p']  
new_label_me = ['attractive_m', 'sincere_partner_m', 'intelligence_m',  
                'funny_m', 'ambition_m', 'shared_interests_m']  
# rating 함수 적용  
for i, j, k in zip(new_label_partner, partner_imp, partner_rate_me):  
    data[i] = data.apply(lambda x: rating(x, j, k), axis=1)  
  
for i, j, k in zip(new_label_me, my_imp, my_rate_partner):  
    data[i] = data.apply(lambda x: rating(x, j, k), axis=1)
```

- 인덱싱을 활용해 총 4개 범주의 변수 이름 리스트를 만든다.
- 계산된 값을 받아줄 새 변수를 생성한다.
- rating() 함수를 적용한다.

#### 4.4 더미 변수 변환

```python
data = pd.get_dummies(data, columns=['gender', 'race', 'race_o'], drop_first=True)
```

### 5. 모델링 및 평가

```python
# 데이터 분할
from sklearn.model_selection import train_test_split  
  
X = data.drop('match', axis=1)  
y = data['match']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)  
  
# 모델링  
import xgboost as xgb  
  
model = xgb.XGBClassifier(n_estimators=500, max_depth=5, random_state=100)  
model.fit(X_train, y_train)  
pred = model.predict(X_test)  
  
# 평가  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  
  
print(accuracy_score(y_test, pred))     # 정확도  
print(confusion_matrix(y_test, pred))   # 혼동 행렬  
print(classification_report(y_test, pred))  # 오류 유형에 따른 평가
```

정확도는 약 **86.16%** 이다. 데이터의 특성을 고려하면 좋지 못한 결과이다.
커플 매칭이 성사된 비율이 약 16%로 나머지 84%는 매칭되지 않았다. 즉, 모든 경우를 매칭되지 않았다고 예측해도 정확도가 84%가 된다는 것이다.

### 6. 하이퍼파라미터 튜닝 : 그리드 서치

**그리드 서치** 를 활요앻 하이퍼파라미터를 튜닝한다.
그리드 서치는 하이퍼파라미터들의 후보를 입력하면 모든 조합에 대해 모델링해보고 최적의 결과가 나오는 조합을 알려준다.

```python
from sklearn.model_selection import GridSearchCV  
  
parameters = {  
    'learning_rate': [0.01, 0.1, 0.3],  
    'max_depth': [5, 7, 10],  
    'subsample': [0.5, 0.7, 1],  
    'n_estimators': [300, 500, 1000]  
}  
  
# 그리드 서치 적용  
model = xgb.XGBClassifier()  
gs_model = GridSearchCV(model, parameters, n_jobs=1, scoring='f1', cv=5)  
gs_model.fit(X_train, y_train)  
print(gs_model.best_params_)  
  
# 평가  
pred = gs_model.predict(X_test)  
print(accuracy_score(y_test, pred))  
print(classification_report(y_test, pred))
```

정확도는 약 **86.34%** 로 미세하게 올라갔다. 일반적으로 하이퍼파라미터 튜닝으로 성능의 엄청난 개선을 하기는 어렵다. 피처 엔지니어링과 모델 알고리즘 선정이 정확도에 가장 큰 영향을 미친다.

### 7. 이해하기 : XGBoost

트리 모델의 진화 과정은 다음과 같다.
- 결정 트리 -> 배깅 -> 랜덤 포레스트 -> 부스팅 -> 경사 부스팅 -> XG 부스팅

#### 7.1 배깅

배깅은 부트스트랩 훈련셋을 사용하는 트리 모델이다. 부트스트랩은 데이터의 일부분을 무작위로 반복 추출하는 방법이다. 데이터의 여러 부분집합을 사용해 여러 트리를 만들어 오버피팅을 방지한다. 이것은 랜덤 포레스트에도 포함된 내용이며 배깅에서 한 단계 발전한 모델이 랜덤 포레스트이다.

#### 7.2 부스팅과 에이다부스트

부스팅과 랜덤 포레스트의 가장 큰 차이는 랜덤 포레스트에서 각 트리는 독립적이나 부스팅은 그렇지 않다는 점이다. 부스팅은 각 트리를 순차적으로 만들면서 이전 트리의 정보를 활용한다. 부분집합을 이용해 첫 트리를 만들고, 해당 트리의 예측 결과를 반영해 두번째 트리를 만들어서 첫번째 트리와의 시너지 효과를 키운다.

부스팅의 대표 알고리즘인 **AdaBoost (Adaptive Boosting)** 은 단계적으로 트리를 만들 때 이전 단계에서의 분류 결과에 따라 각 데이터에 가중치를 부여/수정 한다. 이러한 방식으로 트리 여러 개를 만들면 분류가 복잡한 데이터셋도 세부적으로 나눌 수 있는 모델이 만들어진다.

#### 7.3 경사 부스팅과 XGBoost

**경사 부스팅 (Gradient Boosting)** 은 경사하강법을 이용해 이전 모델의 에러를 기반으로 다음 트리를 만들어간다. 경사 부스팅으로 구현한 알고리즘으로 **XGBoost, LightGBM, CatBoost** 등이 있다.

**XGBoost** 는 계산 성능 최적화와 알고리즘 개선을 함께 이루었기 때문에 특별하다. XGBoost 이전의 부스팅 모델은 트리를 순차적으로 만들기 때문에 모델링 속도가 느리다. XGBoost도 이러한 방식으로 만들지만, 병렬화와 분산 컴퓨팅, 캐시 최적화 등을 활용해 계산 속도가 훨씬 빠르다.
알고리즘을 2차 도함수를 활용해 더 적절한 이동 방향과 크기를 찾아내 더 빠른 시간에 Global Minima 에 도달하도록 개선했다.

또한, 중요한 개선 사항은 **정규화 하이퍼파라미터** 를 지원하는 것이다. 트리 모델이 진화할 수록 오버피팅 문제가 더 심각해질 수 있다. XGBoost는 이러한 부작용을 줄일 목적으로 L1, L2 정규화 하이퍼파라미터를 지원한다.