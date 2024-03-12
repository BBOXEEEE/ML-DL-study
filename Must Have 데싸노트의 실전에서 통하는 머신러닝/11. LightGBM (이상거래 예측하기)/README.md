# 11. LightGBM : 이상거래 예측하기

## 🔥 LightGBM 이란?

**LightGBM** 은 XGBoost 이후로 나온 최신 부스팅 모델이다.
리프 중심 트리 분할 방식을 사용한다.

- 장점
	- XGBoost보다 빠르고 높은 정확도를 보여주는 경우가 많다.
	- 예측에 영향을 미친 변수의 중요도를 확인할 수 있다.
	- 변수 종류가 많고 데이터가 클수록 상대적으로 뛰어난 성능을 보여준다.
- 단점
	- 복잡한 모델인 만큼 해석에 어려움이 있다.
	- 하이퍼파라미터 튜닝이 까다롭다.

## 🔥 카드 거래 내역 데이터셋을 이용한 이상거래 예측하기

### 1. 데이터셋 불러오기

```python
import pandas as pd
  
# 데이터셋 불러오기  
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/fraud.csv'  
data = pd.read_csv(file_url)
```

### 2. 데이터셋 확인

```python
print(data.head())  
print(data.info(show_counts=True))  # Non-Null Count 보이도록 출력  
print(round(data.describe(), 2))
```

- 총 22개의 변수가 있고, Target은 **is_fraud** 이다.

### 3. 전처리 : 데이터 클리닝

Target의 특성을 고려해 불필요한 변수는 제거한다.
- 이름 관련 변수
- street, city, state, zip : 위도, 경도 정보가 있기 때문에 제외
- trans_num : 거래에 대한 id이기 때문에 제외
- unix_time : trans_date_trans_time 변수가 있으므로 제외
- merchant : 상점 관련 정보 제외

```python
data.drop(['first', 'last', 'street', 'city', 'zip', 'trans_num', 'unix_time', 'job', 'merchant'], axis=1, inplace=True)  

data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
```

- 불필요한 변수를 제거하고, `trans_date_trans_time` 변수의 타입을 `datetime` 으로 변경했다.

### 4. 전처리 : 피처 엔지니어링

#### 4.1 amt : 결제 금액

이상거래는 고객의 평소 **소비 패턴** 대비 과도하게 높은 금액을 사용했을 때, 이상거래라고 판단할 수 있다.
이러한 소비 패턴을 파악하기 위해 **Z-Score (표준 점수)** 를 사용한다.

```python
# cc_num별 amt 평균과 표준편차 계산
amt_info = data.groupby('cc_num').agg(['mean', 'std'])['amt'].reset_index()
# 데이터 합치기
data = data.merge(amt_info, on='cc_num', how='left')  
# z-score 계산  
data['amt_z_score'] = (data['amt'] - data['mean']) / data['std']
# 사용하지 않는 변수 제거
data.drop(['mean', 'std'], axis=1, inplace=True)
```

#### 4.2 범주

어떤 범주에 얼만큼의 금액을 사용하는지 또한 개인마다 다를 수 있다.
따라서, 결제 금액과 마찬가지로 Z-Score를 계산하는데, 카드번호와 카테고리 별로 그룹을 지어 계산한다.

```python
# cc_num과 category 기준으로 amt의 평균, 표준편차 계산  
category_info = data.groupby(['cc_num', 'category']).agg(['mean', 'std'])['amt'].reset_index()
# 데이터 합치기
data = data.merge(category_info, on=['cc_num', 'category'], how='left')
# z-score 계산  
data['cat_z_score'] = (data['amt'] - data['mean']) / data['std']
# 사용하지 않는 변수 제거
data.drop(['mean', 'std'], axis=1, inplace=True)   
```

#### 4.3 거리

고객의 위치와 상점의 위치 변수가 있기 때문에 둘 사이의 거리를 구할 수 있다.
거리에 대한 Z-Score를 통해 기존 패턴에서 벗어난 거래를 감지할 수 있다.
즉, 주소지 인근이 아닌 멀리 떨어진 지역에서 거래가 발생했다면 타인이 사용한 것인지 의심해 볼 여지가 있기 때문이다.
거리 정보를 계산하는데 **geopy** 라이브러리를 사용한다.

```python
import geopy.distance  

# 위도, 경도 한 변수로 합치기 
data['merch_coord'] = pd.Series(zip(data['merch_lat'], data['merch_long']))  
data['cust_coord'] = pd.Series(zip(data['lat'], data['long']))  
# 거리 계산  
data['distance'] = data.apply(lambda x: geopy.distance.distance(x['merch_coord'], x['cust_coord']).km, axis=1)  
  
# cc_num 별, 거리 정보 계산  
distance_info = data.groupby('cc_num').agg(['mean', 'std'])['distance'].reset_index()
# 데이터 합치기
data = data.merge(distance_info, on='cc_num', how='left')  
# z-score 계산  
data['distance_z_score'] = (data['distance'] - data['mean']) / data['std']  
# 사용하지 않는 변수 제거
data.drop(['mean', 'std'], axis=1, inplace=True)    
```

#### 4.4 나이

생년월일 정보를 활용해 나이를 계산한다.

```python
data['age'] = 2024 - pd.to_datetime(data['dob']).dt_year  
```

또한, 전처리 후 사용하지 않는 변수들을 제거한다.

```python
# 전처리 후 사용하지 않는 변수 제거  
data.drop(['cc_num', 'lat', 'long', 'merch_lat', 'merch_long', 'dob',
		   'merch_coord', 'cust_coord'], axis=1, inplace=True)
```

#### 4.5 더미 변수 변환

```python
# 더미 변수 변환  
data = pd.get_dummies(data, columns=['category', 'gender'], drop_first=True)  
# trans_date_trans_time 은 인덱스로 설정  
data.set_index('trans_date_trans_time', inplace=True)
```

### 5. 모델링 및 평가

데이터셋을 랜덤하게 분할하지 않고, 특정 시점을 기준으로 분할한다.
이상거래를 감지하는 상황은 현재까지 발생한 거래 데이터를 기반으로 앞으로 일어나는 거래에 대한 예측을 해야하기 때문이다.

```python
# 데이터 분할  
train = data[data.index < '2020-07-01']  
test = data[data.index >= '2020-07-01']  
print(len(test) / len(data))  
  
X_train = train.drop('is_fraud', axis=1)  
X_test = test.drop('is_fraud', axis=1)  
y_train = train['is_fraud']  
y_test = test['is_fraud']  
  
# 모델링  
import lightgbm as lgb  
  
model_1 = lgb.LGBMClassifier(random_state=100)  
model_1.fit(X_train, y_train)  
pred_1 = model_1.predict(X_test)
```

평가 방법으로 정확도, 혼동 행렬, 분류 리포트 그리고 **ROC AUC 점수** 까지 활용한다.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score  
  
print(accuracy_score(y_test, pred_1))   # 정확도  
print(confusion_matrix(y_test, pred_1))     # 혼동 행렬  
print(classification_report(y_test, pred_1))    # 분류 리포트  
  
proba_1 = model_1.predict(X_test)  
proba_1 = proba_1[:, 1]  
print(roc_auc_score(y_test, proba_1))   # 정확도 확인
```

- `Classification Report` 에서 정밀도와 재현율, F1 점수 중 재현율이 더 중요하다.
	- 실제 이상거래를 얼마나 많이 예측했는지를 의미하기 때문이다.
- `AUC` 라는 지표를 이해하기 위해서 `ROC 곡선` 에 대한 이해가 필요하다.
	- 민감도와 특이도 개념을 활용한다.
		- 민감도는 실제 1인 것 중 얼만큼 제대로 1로 예측되었는지를 의미하며, 1에 가까울수록 좋은 수치이다.
		- 특이도는 실제 0인 것 중 얼만큼 1로 잘못 예측되었는지를 의미하며, 0에 가까울수록 좋은 수치이다.
	- AUC 점수는 보통 0.8 이상이면 상당히 높은 편이다.
		- 현재 0.9031 정도로 높은 편이지만, 데이터가 편향되었다면 자연스럽게 높게 나오는 경향이 있다.
		- 현재 데이터는 이상거래는 1% 비율로, 편향되어 있다.

### 6. 하이퍼파라미터 튜닝 : 랜덤 그리드 서치

기존 그리드 서치와 달리 하이퍼파라미터들의 조합을 랜덤으로 일부만 선택하여 모델링하는 **랜덤 그리드 서치** 를 통해 최적의 하이퍼파라미터를 찾는다.

```python
from sklearn.model_selection import RandomizedSearchCV  
  
params = {  
    'n_estimators': [100, 500, 1000],  
    'learning_rate': [0.01, 0.05, 0.1, 0.3],  
    'lambda_l1': [0, 10, 20, 30, 50],  
    'lambda_l2': [0, 10, 20, 30, 50],  
    'max_depth': [5, 10, 15, 20],  
    'subsample': [0.6, 0.8, 1]  
}  
  
# 랜덤 그리드 서치 적용  
model_2 = lgb.LGBMClassifier(random_state=100)  
rs = RandomizedSearchCV(model_2, param_distributions=params, n_iter=30,
						scoring='roc_auc', random_state=100, n_jobs=1)  
  
# 시간 측정  
import time  
  
start = time.time()  
rs.fit(X_train, y_train)  
print(time.time() - start)  
  
# 결과  
print(rs.best_params_)  
rs_proba = rs.predict(X_test)  
print(roc_auc_score(y_test, rs_proba[:, 1]))
```

결과적으로 약 0.995로 이전 대비 크게 좋아진 것을 확인할 수 있고, 거짓 양성 비율이 훨씬 좋아졌다.