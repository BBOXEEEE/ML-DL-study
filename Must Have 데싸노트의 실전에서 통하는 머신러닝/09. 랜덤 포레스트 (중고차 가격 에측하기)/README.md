# 09. 랜덤 포레스트 : 중고차 가격 예측하기

## 🔥 Random Forest 란?

**Random Forest** 모델을 결정 트리의 단점인 오버피팅 문제를 완화시켜주는 발전된 형태의 트리 모델이다.
랜덤으로 생성된 무수히 많은 트리를 이용해 예측하기 때문에 이와 같이 불리며 여러 모델을 활용해 하나의 모델을 이루는 기법을 **앙상블** 이라고 부른다.

- 장점
	- 결정 트리와 마찬가지로 아웃라이어에 거의 영향을 받지 않는다.
	- 선형/비선형 데이터에 상관없이 잘 작동한다.
- 단점
	- 학습 속도가 상대적으로 느린 편이다.
	- 수많은 트리를 동원하기 때문에 모델에 대한 해석이 어렵다.

## 🔥 중고차 판매 이력 데이터셋을 이용한 중고차 가격 예측하기

### 1. 데이터셋 불러오기

``` python
import pandas as pd  
  
# 데이터셋 불러오기  
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/car.csv'  
data = pd.read_csv(file_url)
```

### 2. 데이터셋 확인하기

```python
print(data.head())  
print(data.info())  
print(round(data.describe(), 2))
```

데이터의 컬럼들은 다음과 같다.

- Feature
	- name : 이름
	- year : 생산년
	- km_driven : 주행거리
	- fuel : 연료
	- seller_type : 판매자 유형
	- transmission : 변속기
	- owner : 차주 변경 내역
	- mileage : 마일리지
	- engine : 배기량
	- max_power : 최대 출력
	- torque : 토크
	- seats : 인승
- Target
	- selling_price : 판매가

데이터 정보를 확인해보면, 결측치도 있고 object 변수도 있다.

### 3. 전처리 : 텍스트 데이터

문자형 데이터를 숫자형으로 바꾸어야 연산이 가능하다.
첫번째 작업으로 단위 일치 및 숫자형으로 변환을 진행한다.
두번째는 불필요한 텍스트는 버리고 필요한 부분만 남긴다.

#### 3.1 engine

`pandas` 시리즈에서 제공하는 **str.split()** 을 이용한다.
데이터 프레임이 아닌 시리즈에만 있는 메소드이므로 컬럼 하나씩만 인덱싱 해준다.

```python
data[['engine', 'engine_unit']] = data['engine'].str.split(expand=True)  
data['engine'] = data['engine'].astype('float32')  
print(data['engine'].head())  
data.drop('engine_unit', axis=1, inplace=True)
```

- 숫자와 문자를 분리해 `engine` 과 `engine_unit` 에 저장했다.
- 데이터 타입을 **float32** 로 변환했다.
- 필요없는 데이터인 `engine_unit` 은 **drop** 한다.

#### 3.2 max_power

```python
data[['max_power', 'max_power_unit']] = data['max_power'].str.split(expand=True)
```

- `engine` 과 같은 방식으로 데이터 타입을 변환하게 되면 **ValueError** 가 발생한다.
- 원래 데이터에 숫자 없이 문자만 있어 분리되지 않은 것이다.
- 이를 해결하기 위해 `Try-Except` 블록을 별도 메소드로 만들어 사용할 수 있다.

```python
def isFloat(value):  
    try:  
        num = float(value)  
        return num  
    except ValueError:  
        return np.NaN  
  
  
data['max_power'] = data['max_power'].apply(isFloat)  
print(data['max_power'].head())  
data.drop('max_power_unit', axis=1, inplace=True)
```

- 위와 같이 에러가 발생하는 값은 **NaN** 으로 채운다.

#### 3.3 mileage

```python
data[['mileage', 'mileage_unit']] = data['mileage'].str.split(expand=True)  
data['mileage'] = data['mileage'].astype('float32')
```

- 먼저 숫자만 남은 `mileage` 는 **float32** 로 데이터 타입을 변환한다.

`mileage_unit` 의 고유값은 단위가 있는 것을 확인할 수 있다.
 - `kmpl`  : 리터 당 킬로미터
 - `km/kg` : 킬로그램 당 킬로미터

단위를 통일하기 위해 리터 당 연료의 가격을 활용한다.

```python
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
```

- `mileage` 변수를 각 연료별 가격으로 나누면 **1달러 당 주행거리** 가 된다.

#### 3.4 torque

`torque` 변수의 값은 처리해야 할 과정이 조금 길다.

**190Nm@ 2000rpm** 과 같은 데이터를 다음 과정을 거쳐 전처리한다.
 - 앞부분의 숫자만 추출해 숫자형으로 변환한다.
 - Nm 단위로 스케일링 한다.

```python
data['torque'] = data['torque'].str.upper()  # torque 변수 대문자로 변환  
  
  
def torque_unit(x):  # 단위를 분류하는 함수  
    if 'NM' in str(x):  
        return 'Nm'  
    elif 'KGM' in str(x):  
        return 'kgm'  
  
  
data['torque_unit'] = data['torque'].apply(torque_unit)  
data['torque_unit'].fillna('Nm', inplace=True)  # 결측치를 Nm으로 대체
```

- `torque` 변수를 모두 대문자로 변환한다.
- 결측치를 모두 **Nm** 로 대체한다.

``` python
def split_num(x):  # 숫자 분리 함수  
    x = str(x)  
    cut = 0  
    for i, j in enumerate(x):  
        if j not in '0123456789.':  
            cut = i  
            break  
    return x[:cut]  
  
  
data['torque'] = data['torque'].apply(split_num)  # 숫자만 빼내기  
data['torque'] = data['torque'].replace('', np.NaN)  # ''를 결측치로 대체  
data['torque'] = data['torque'].astype('float64')  # 데이터 타입 변환  
print(data['torque'].head())
```

- 숫자만 빼내는 함수를 만들고 숫자만 `torque` 에 저장한다.
- 위 예제와 비슷하게 바로 데이터 타입을 변환하면 에러가 발생한다.
- 결측치를 **NaN** 으로 대체 후 데이터 타입을 변환한다.

```python
def torque_trans(x):  # 단위의 차이 맞추는 함수  
    if x['torque_unit'] == 'kgm':  
        return x['torque'] * 9.8066  
    else:  
        return x['torque']  
  
  
data['torque'] = data.apply(torque_trans, axis=1)  
data.drop('torque_unit', axis=1, inplace=True)
```

- `kgm * 9.8066 = Nm` 이므로 단위에 의한 차이를 맞춰준다.
- 이제 더이상 필요하지 않은 `torque_unit` 은 **drop** 한다.

#### 3.5 name

`name` 변수에는 브랜드 이름과 모델명이 담겨있다.
자동차 가격을 예측하는데 자동차의 특성이 더 중요하므로 수많은 자동차 모델을 다 담아내는 것은 비효율적이다.
다만, 같은 스펙이더라도 비싼 브랜드의 자동차 가격이 더 비싸기 때문에 브랜드는 유지하는 것이 좋다.
따라서, 브랜드 명만 분리해 유지한다.

```python
data['name'] = data['name'].str.split(expand=True)[0]  
print(data['name'].unique())  
data['name'] = data['name'].replace('Land', 'Land Rover')
```

- 이때, 띄어쓰기로 인해 분리된 `Land` 만 `Land Rover` 로 치환해주었다.

### 4. 전처리 : 결측치 처리와 더미 변수 변환

```python
print(data.isna().mean())  
# 결측치가 있는 행 삭제  
data.dropna(inplace=True)  

# 더미 변수 변환  
data = pd.get_dummies(data, columns=['name', 'fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
```

- 결측치를 평균값으로 채우는 것은 노이즈 역할을 할 가능성이 높다. 또한, 결측치 비율이 약 2% 수준으로 높지 않기 때문에 결측치가 있는 행은 모두 삭제한다.

### 5. 모델링 및 평가

```python
# 데이터 분리  
from sklearn.model_selection import train_test_split  
  
X = data.drop('selling_price', axis=1)  
y = data['selling_price']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)  
  
# 모델링  
from sklearn.ensemble import RandomForestRegressor  
  
model = RandomForestRegressor(random_state=100)  
model.fit(X_train, y_train)  
train_pred = model.predict(X_train)  
test_pred = model.predict(X_test)  
  
# 평가  
from sklearn.metrics import mean_squared_error  
  
print("train_rmse: ", mean_squared_error(y_train, train_pred) ** 0.5,  
      "test_rmse: ", mean_squared_error(y_test, test_pred) ** 0.5)
```

결과는 다음과 같다.
- **train_rmse:  53531.41548125947 test_rmse:  131855.18391308116**

### 6. K-Fold Cross Validation

**Corss Validation** 의 목적은 모델의 예측력을 더 안정적으로 평가하기 위함이다.
random sampling 으로 나누어졌더라도 우연에 의한 오차들이 예측력을 평가하는데 작은 노이즈로 존재한다.

**K-Fold Cross Validation** 은 데이터를 특정 개수 (K개)로 쪼개어서 그중 하나씩을 선택해 시험셋으로 사용하되, 이 과정을 K번만큼 반복하는 것이다.

```python
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
```

K-Fold Cross Validation 을 적용한 결과는 다음과 같다.
- **train_rmse:  56553.836119114814 test_rmse:  142936.58918244042**
- 앞의 결과에 비해 RMSE가 높아졌지만 교차검증을 사용한 결과가 조금 더 정확한 평가 결과이다.

### 7. 하이퍼파라미터 튜닝

랜덤 포레스트는 수많은 하이퍼파라미터를 가지고 있다.

- **n_estimate** : 결정 트리의 개수이다. 기본값은 100이며, 너무 많거나 적은 수를 입력하면 성능이 떨어진다.
- **max_depth** : 각 트리의 최대 깊이를 제한한다.
- **min_samples_split** : 해당 노드를 나눌 것인지 말 것인지를 노드 데이터 수를 기준으로 판단한다. 지정된 숫자보다 적은 수의 데이터가 노드에 있으면 더는 분류하지 않는다.
- **min_samples_leaf** : 분리된 노드의 데이터에 최소 몇 개의 데이터가 있어야 할지를 결정한다. 지정된 숫자보다 적은 수의 데이터가 분류되면 해당 분리는 이루어지지 않는다.
- **n_jobs** : 병렬 처리에 사용되는 CPU 코어 수이다.

임의의 숫자를 넣은 예시와 결과는 다음과 같다.

```python
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
```

- **train_rmse:  66762.84568886801 test_rmse:  142205.83441414658**
- test RMSE 는 조금 더 낮아졌다. 조금이나마 오버피팅이 줄어들었다고 볼 수 있다.
