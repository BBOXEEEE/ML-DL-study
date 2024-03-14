import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 인위적으로 만든 데이터셋
# 데이터셋 불러오기
# file_url = 'https://raw.githubusercontent.com/musthave-ML10/data_source/main/example_cluster.csv'
# data = pd.read_csv(file_url)
# data.to_csv('./_data/example_cluster.csv', index=False)
file_path = './_data/example_cluster.csv'
data = pd.read_csv(file_path)

# 데이터셋 확인
print(data)
sns.scatterplot(x='var_1', y='var_2', data=data)
# plt.show()

print('#' * 50, 1)
print()

# 연습용 데이터 모델링 및 평가
from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters=3, random_state=100)
kmeans_model.fit(data)
data['label'] = kmeans_model.predict(data)

# 산점도 확인
sns.scatterplot(x='var_1', y='var_2', data=data, hue='label', palette='rainbow')
# plt.show()

# 엘보우 기법으로 최적의 K값 구하기
distance = []
for k in range(2, 10):
    k_model = KMeans(n_clusters=k)
    k_model.fit(data)
    distance.append(k_model.inertia_)

print(distance)
sns.lineplot(x=range(2, 10), y=distance)
# plt.show()

print('#' * 50, 2)
print()

# 2. 고객 데이터셋 활용
# 데이터셋 불러오기
# file_url = 'https://raw.githubusercontent.com/musthave-ML10/data_source/main/customer.csv'
# customer = pd.read_csv(file_url)
# customer.to_csv('./_data/customer.csv')
file_path = './_data/customer.csv'
customer = pd.read_csv(file_path)

# 데이터셋 확인
print(data.head())
print(customer['cc_num'].nunique())     # 고유값 확인 : 고객의 수
print(customer['category'].nunique())   # 고유값 확인 : 범주의 수

print('#' * 50, 3)
print()

# 전처리 : 피처 엔지니어링
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

print('#' * 50, 4)
print()

# 모델링 및 실루엣 계수
# (1) 엘보우 기법으로 최적의 K값 찾기
distance = []
for k in range(2, 10):
    k_model = KMeans(n_clusters=k)
    k_model.fit(scaled_df)
    labels = k_model.predict(scaled_df)
    distance.append(k_model.inertia_)

sns.lineplot(x=range(2, 10), y=distance)
# plt.show()

# (2) 실루엣 계수
from sklearn.metrics import silhouette_score

silhouette = []
for k in range(2, 10):
    k_model = KMeans(n_clusters=k)
    k_model.fit(scaled_df)
    labels = k_model.predict(scaled_df)
    silhouette.append(silhouette_score(scaled_df, labels))

sns.lineplot(x=range(2, 10), y=silhouette)
plt.show()

print('#' * 50, 5)
print()

# 최종 예측 모델 및 결과 해석
# (1) 모델링
k_model = KMeans(n_clusters=4)
k_model.fit(scaled_df)
labels = k_model.predict(scaled_df)
scaled_df['label'] = labels

# (2) label 별 데이터 요약
scaled_df_mean = scaled_df.groupby('label').mean()  # label 별 평균값
scaled_df_count = scaled_df.groupby('label').count()['category_travel']     # label 별 등장 횟수
scaled_df_count = scaled_df_count.rename('count')   # 변수 이름 수정
scaled_df_all = scaled_df_mean.join(scaled_df_count)    # 데이터 합치기
print(scaled_df_all)
