import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 불러오기
# file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/fraud.csv'
# data = pd.read_csv(file_url)
# data.to_csv('./_data/fraud.csv', index=False)
file_path = './_data/fraud.csv'
data = pd.read_csv(file_path)

# 데이터셋 확인
print(data.head())
print(data.info(show_counts=True))  # Non-Null Count 보이도록 출력
print(round(data.describe(), 2))

print('#' * 50, 1)
print()

# 전처리 : 데이터 클리닝
# 불필요한 변수 제거
data.drop(['first', 'last', 'street', 'city', 'zip', 'trans_num', 'unix_time', 'job', 'merchant'], axis=1, inplace=True)
# 날짜 형식으로 변환
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

# 전처리 : 피처 엔지니어링
# (1) amt (결제 금액)
# cc_num별 amt 평균과 표준편차 계산
amt_info = data.groupby('cc_num').agg(['mean', 'std'])['amt'].reset_index()
data = data.merge(amt_info, on='cc_num', how='left')
data['amt_z_score'] = (data['amt'] - data['mean']) / data['std']    # z-score 계산
data.drop(['mean', 'std'], axis=1, inplace=True)    # 사용하지 않는 변수 제거

# (2) 범주
# cc_num과 category 기준으로 amt의 평균, 표준편차 계산
category_info = data.groupby(['cc_num', 'category']).agg(['mean', 'std'])['amt'].reset_index()
data = data.merge(category_info, on=['cc_num', 'category'], how='left')
data['cat_z_score'] = (data['amt'] - data['mean']) / data['std']    # z-score 계산
data.drop(['mean', 'std'], axis=1, inplace=True)    # 사용하지 않는 변수 제거

# (3) 거리
import geopy.distance
data['merch_coord'] = pd.Series(zip(data['merch_lat'], data['merch_long'])) # 위도, 경도 한 변수로 합치기
data['cust_coord'] = pd.Series(zip(data['lat'], data['long']))
# 거리 계산
data['distance'] = data.apply(lambda x: geopy.distance.distance(x['merch_coord'], x['cust_coord']).km, axis=1)

# cc_num 별, 거리 정보 계산
distance_info = data.groupby('cc_num').agg(['mean', 'std'])['distance'].reset_index()
data = data.merge(distance_info, on='cc_num', how='left')
data['distance_z_score'] = (data['distance'] - data['mean']) / data['std']  # z-score 계산
data.drop(['mean', 'std'], axis=1, inplace=True)    # 사용하지 않는 변수 제거

# (4) 나이 구하기
data['age'] = 2024 - pd.to_datetime(data['dob']).dt_year
# 전처리 후 사용하지 않는 변수 제거
data.drop(['cc_num', 'lat', 'long', 'merch_lat', 'merch_long', 'dob', 'merch_coord', 'cust_coord'],
          axis=1, inplace=True)

# (5) 새 변수 만들기
# 더미 변수 변환
data = pd.get_dummies(data, columns=['category', 'gender'], drop_first=True)
# trans_date_trans_time 은 인덱스로 설정
data.set_index('trans_date_trans_time', inplace=True)

print(data.head())

print('#' * 50, 2)
print()

# 모델링 및 평가
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

# 평가
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

print(accuracy_score(y_test, pred_1))   # 정확도
print(confusion_matrix(y_test, pred_1))     # 혼동 행렬
print(classification_report(y_test, pred_1))    # 분류 리포트

proba_1 = model_1.predict(X_test)
proba_1 = proba_1[:, 1]
print(roc_auc_score(y_test, proba_1))   # 정확도 확인

print('#' * 50, 3)
print()

# 하이퍼파라미터 튜닝 : 랜덤 그리드 서치
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
rs = RandomizedSearchCV(model_2, param_distributions=params, n_iter=30, scoring='roc_auc',
                        random_state=100, n_jobs=1)

# 시간 측정
import time

start = time.time()
rs.fit(X_train, y_train)
print(time.time() - start)

# 결과
print(rs.best_params_)
rs_proba = rs.predict(X_test)
print(roc_auc_score(y_test, rs_proba[:, 1]))
