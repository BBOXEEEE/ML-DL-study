import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 차원 축소해 그래프 그리기 : 고객 데이터셋
# 데이터셋 불러오기
file_url = 'https://raw.githubusercontent.com/musthave-ML10/data_source/main/customer_pca.csv'
customer = pd.read_csv(file_url)

# 데이터셋 확인
print(customer.head())
# 독립 변수, 종속 변수 분리
customer_X = customer.drop('label', axis=1)
customer_y = customer['label']

print('#' * 50, 1)
print()

# 그래프 표현을 위한 차원 축소
from sklearn.decomposition import PCA

pca = PCA(n_components=2)   # n_components : 주성분 개수
pca.fit(customer_X)
customer_pca = pca.transform(customer_X)
customer_pca = pd.DataFrame(customer_pca, columns=['PC1', 'PC2'])
customer_pca = customer_pca.join(customer_y)

# 최종 데이터 확인
print(customer_pca.head())

# 산점도 그래프
sns.scatterplot(x='PC1', y='PC2', data=customer_pca, hue='label', palette='rainbow')
plt.show()

# 주성분과 변수의 관계
df_comp = pd.DataFrame(pca.components_, columns=customer_X.columns)
sns.heatmap(df_comp, cmap='coolwarm')
plt.show()

print('#' * 50, 2)
print()

# 2. 속도와 예측력 향상시키기 : 익명 데이터셋
# 데이터셋 불러오기
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/anonymous.csv'
anonymous = pd.read_csv(file_url)

# 데이터셋 확인
print(anonymous.head())
print(anonymous['class'].mean())    # 종속변수의 평균 확인
print(anonymous.isna().sum().sum())     # 결측치 확인

# PCA에 따른 모델링 성능/결과 비교하기
# (1) 데이터 분할
from sklearn.model_selection import train_test_split

X = anonymous.drop('class', axis=1)
y = anonymous['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# (2) 스케일링
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# (3) 모델링
from sklearn.ensemble import RandomForestClassifier

model_1 = RandomForestClassifier(random_state=100)
model_1.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score, roc_auc_score

pred_1 = model_1.predict(X_test_scaled)
print(accuracy_score(y_test, pred_1))
proba_1 = model_1.predict_proba(X_test_scaled)
print(roc_auc_score(y_test, proba_1[:, 1]))

# (4) PCA
pca = PCA(n_components=2)
pca.fit(X_train_scaled)
print(pca.explained_variance_ratio_)    # 데이터 반영 비율 확인

# 주성분 개수 찾기
var_ratio = []
for i in range(100, 550, 50):
    pca = PCA(n_components=i)
    pca.fit_transform(X_train_scaled)
    ratio = pca.explained_variance_ratio_.sum()
    var_ratio.append(ratio)

# 그래프로 확인
sns.lineplot(x=range(100, 550, 50), y=var_ratio)
plt.show()

# 주성분이 400개인 데이터 만들기
pca = PCA(n_components=400, random_state=100)
pca.fit(X_train_scaled)
X_train_scaled_pca = pca.transform(X_train_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)

print('#' * 50, 3)
print()

# 랜덤 포레스트
model_2 = RandomForestClassifier(random_state=100)
model_2.fit(X_train_scaled_pca, y_train)

pred_2 = model_2.predict(X_test_scaled_pca)
print(accuracy_score(y_test, pred_2))
proba_2 = model_2.predict_proba(X_test_scaled_pca)
print(roc_auc_score(y_test, proba_2[:, 1]))
