import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 불러오기
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/car.csv'
data = pd.read_csv(file_url)

# 데이터셋 확인
print(data.head())
print(data.info())
print(round(data.describe(), 2))

print('#' * 50, 1)
print()

# 전처리 : 텍스트 데이터
# (1) engine
data[['engine', 'engine_unit']] = data['engine'].str.split(expand=True)
data['engine'] = data['engine'].astype('float32')
print(data['engine'].head())
data.drop('engine_unit', axis=1, inplace=True)

print()

# (2) max_power
data[['max_power', 'max_power_unit']] = data['max_power'].str.split(expand=True)


def isFloat(value):
    try:
        num = float(value)
        return num
    except ValueError:
        return np.NaN


data['max_power'] = data['max_power'].apply(isFloat)
print(data['max_power'].head())
data.drop('max_power_unit', axis=1, inplace=True)

print()

# (3) mileage
data[['mileage', 'mileage_unit']] = data['mileage'].str.split(expand=True)
data['mileage'] = data['mileage'].astype('float32')


def mile(x):
    if x['fuel'] == 'Petrol':
        return x['mileage'] / 80.43
    elif x['fuel'] == 'Diesel':
        return x['mileage'] / 73.56
    elif x['fuel'] == 'LPG':
        return x['mileage'] / 40.85
    else:
        return x['mileage'] / 44.23


data['mileage'] = data.apply(mile, axis=1)
data.drop('mileage_unit', axis=1, inplace=True)

print()

# (4) torque
data['torque'] = data['torque'].str.upper()  # torque 변수 대문자로 변환


def torque_unit(x):  # 단위를 분류하는 함수
    if 'NM' in str(x):
        return 'Nm'
    elif 'KGM' in str(x):
        return 'kgm'


data['torque_unit'] = data['torque'].apply(torque_unit)
data['torque_unit'].fillna('Nm', inplace=True)  # 결측치를 Nm으로 대체


def split_num(x):  # 숫자 분리 함수
    x = str(x)
    cut = 0
    for i, j in enumerate(x):
        if j not in '0123456789.':
            cut = i
            break
    return x[:cut]


data['torque'] = data['torque'].apply(split_num)  # 숫자만 빼내기
print(data['torque'])
data['torque'] = data['torque'].replace('', np.NaN)  # ''를 결측치로 대체
data['torque'] = data['torque'].astype('float64')  # 데이터 타입 변환
print(data['torque'].head())


def torque_trans(x):  # 단위의 차이 맞추는 함수
    if x['torque_unit'] == 'kgm':
        return x['torque'] * 9.8066
    else:
        return x['torque']


data['torque'] = data.apply(torque_trans, axis=1)
data.drop('torque_unit', axis=1, inplace=True)

print()

# (5) name
data['name'] = data['name'].str.split(expand=True)[0]
print(data['name'].unique())
data['name'] = data['name'].replace('Land', 'Land Rover')

print('#' * 50, 2)
print()

# 전처리 : 결측치 처리와 더미 변수 변환
# (1) 결측치 확인
print(data.isna().mean())

print()

# (2) 결측치가 있는 행 삭제
data.dropna(inplace=True)
print(len(data))

print()

# (3) 더미 변수 변환
data = pd.get_dummies(data, columns=['name', 'fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
print(data.head())

print('#' * 50, 3)
print()

# 모델링 및 평가
# (1) 데이터 분리
from sklearn.model_selection import train_test_split

X = data.drop('selling_price', axis=1)
y = data['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# (2) 모델링
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=100)
model.fit(X_train, y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# (3) 평가
from sklearn.metrics import mean_squared_error

print("train_rmse: ", mean_squared_error(y_train, train_pred) ** 0.5,
      "test_rmse: ", mean_squared_error(y_test, test_pred) ** 0.5)

print('#' * 50, 4)
print()

# K-Fold cross validation
from sklearn.model_selection import KFold

# 데이터 인덱스 정리
data.reset_index(drop=True, inplace=True)

# K-Fold
kf = KFold(n_splits=5)
X = data.drop('selling_price', axis=1)
y = data['selling_price']

train_rmse_total = []
test_rmse_total = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    model = RandomForestRegressor(random_state=100)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5

    train_rmse_total.append(train_rmse)
    test_rmse_total.append(test_rmse)

print("train_rmse: ", sum(train_rmse_total) / 5,
      "test_rmse: ", sum(test_rmse_total) / 5)

print('#' * 50, 5)
print()

# 하이퍼파라미터 튜닝
train_rmse_total = []
test_rmse_total = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    model = RandomForestRegressor(n_estimators=300, max_depth=50, min_samples_split=5,
                                  min_samples_leaf=1, n_jobs=1, random_state=100)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5

    train_rmse_total.append(train_rmse)
    test_rmse_total.append(test_rmse)

print("train_rmse: ", sum(train_rmse_total) / 5,
      "test_rmse: ", sum(test_rmse_total) / 5)
