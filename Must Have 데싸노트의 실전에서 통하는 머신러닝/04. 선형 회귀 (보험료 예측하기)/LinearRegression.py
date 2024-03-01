import pandas as pd

# 데이터 불러오기
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/insurance.csv'
data = pd.read_csv(file_url)

# 데이터셋의 다양한 정보 확인하기
print(data)
print(data.info())
print(round(data.describe()), 2)

print('#' * 100)

# 전처리 : train_dataset, test_dataset 나누기
X = data[['age', 'sex', 'bmi', 'children', 'smoker']]
y = data['charges']

# 데이터 스플릿을 위한 라이브러리 import
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# 모델링
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# 예측하기
pred = model.predict(X_test)

# 평가하기
comparison = pd.DataFrame({'actual': y_test, 'pred': pred})

print(comparison)

print('#' * 100)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 10))
sns.scatterplot(x='actual', y='pred', data=comparison)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, pred) ** 0.5)

model.score(X_train, y_train)
