import pandas as pd

# 데이터 불러오기
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/titanic.csv'
data = pd.read_csv(file_url)

# 데이터셋 정보 출력
print(data.head())
print(data.info())
print(data.describe())
# print(data.corr())

import matplotlib.pyplot as plt
import seaborn as sns

# 상관관계 분석
#sns.heatmap(data.corr())
#plt.show()
#sns.heatmap(data.corr(), cmap='coolwarm', vmin=-1, vmax=1)
#sns.heatmap(data.corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=True)

print('#' * 100)

# 범주형 변수 변환 (one-hot encoding)
# 고유값 개수 확인
print(data['Name'].nunique())       # 889
print(data['Sex'].nunique())        # 2
print(data['Ticket'].nunique())     # 680
print(data['Embarked'].nunique())   # 3

# 학습에 필요없는 변수 제거
data = data.drop(['Name', 'Ticket'], axis=1)
print(data.head())

# one-hot encoding
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
print(data.head())

# 모델링 및 예측하기
from sklearn.model_selection import train_test_split

# data split
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# 모델 import
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
pred = model.predict(X_test)

print('#' * 100)

# 예측 결과 평가
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))

# Logistic Regression model 의 계수 확인
print(model.coef_)
print(pd.Series(model.coef_[0], index=X.columns))

print('#' * 100)

# Feature Engineering
data['Family'] = data['SibSp'] + data['Parch']
data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
print(data.head())

# 재학습
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(accuracy_score(y_test, pred))
