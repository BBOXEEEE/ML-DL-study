import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/wine.csv'
data = pd.read_csv(file_url)

# 데이터셋 정보 확인
print(data.head())
print(data.info())
print(data.describe())

print("#" * 50, 1)

# Target의 고유값 확인
print(data['class'].unique())
print(data['class'].value_counts())  # 각 고유값에 해당하는 개수 출력
sns.barplot(x=data['class'].value_counts().index, y=data['class'].value_counts())
# plt.show()

print('#' * 50, 2)

# 데이터 전처리 : 결측치 처리
# 결측치 여부를 다양한 방법으로 확인
print(data.isna())  # 결측치 여부 출력
print(data.sum())  # 변수별 합 출력
print(data.isna().sum())  # isna() 와 sum() 을 조합해 결측치 확인
print(data.isna().mean())  # 결측치 비율 출력

# 방식1. 결측치 행 제거
print(data.dropna().isna().mean())

# 방식2. 결측 변수 제거
print(data.drop(['alcohol', 'nonflavanoid_phenols'], axis=1))

# 방식3. 결측값 채우기
print(data.fillna(-99))  # -99로 결측치 채우기
print(data.fillna(data.mean()))  # 평균값으로 결측치 채우기

# 최종 결측치 처리 : 중간값 사용
data.fillna(data.median(), inplace=True)
print(data.isna().mean())

print('#' * 50, 3)

# 스케일링
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 1. 표준화 스케일링
st_scaler = StandardScaler()
st_scaler.fit(data)
st_scaled = st_scaler.transform(data)
# print(pd.DataFrame(st_scaled))
st_scaled = pd.DataFrame(st_scaled, columns=data.columns)
print(st_scaled)

# 2. 로버스트 스케일링
rb_scaler = RobustScaler()
rb_scaled = rb_scaler.fit_transform(data)
rb_scaled = pd.DataFrame(rb_scaled, columns=data.columns)
print(rb_scaled)

# 3. 최소-최대 스케일링
mm_scaler = MinMaxScaler()
mm_scaled = mm_scaler.fit_transform(data)
mm_scaled = pd.DataFrame(mm_scaled, columns=data.columns)
print(mm_scaled)

print('#' * 50, 4)

# 스케일링 적용
from sklearn.model_selection import train_test_split
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# 최소-최대 스케일링 적용
mm_scaler = MinMaxScaler()
mm_scaler.fit(X_train)
X_train_scaled = mm_scaler.transform(X_train)
X_test_scaled = mm_scaler.transform(X_test)

# 모델링 및 예측/평가
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
pred = knn.predict(X_test_scaled)
print(accuracy_score(y_test, pred))

print('#' * 50, 5)

# 하이퍼파라미터 튜닝
# 1. n_neighbors=7 로 parameter 조정
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
pred = knn.predict(X_test_scaled)
print(accuracy_score(y_test, pred))

# 2. n_neighbors=3 으로 paramter 조정
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
pred = knn.predict(X_test_scaled)
print(accuracy_score(y_test, pred))

print('#' * 50, 6)

# 3. 반복 확인
scores = []
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, y_train)
    pred = knn.predict(X_test_scaled)
    scores.append(accuracy_score(y_test, pred))

print(scores)

# 그래프로 출력
sns.lineplot(x=range(1, 21), y=scores)
#plt.show()

print('#' * 50, 7)

# n_neighbors = 13 으로 확정
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train_scaled, y_train)
pred = knn.predict(X_test_scaled)
print(accuracy_score(y_test, pred))
