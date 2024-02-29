# 03. 유용한 라이브러리 : 판다스와 넘파이

## 🔥 Pandas

`Pandas` 는 데이터 분석에서 가장 많이 쓰이는 라이브러리이다. 주로 **DataFrame** 과 **Series** 라는 두 가지 자료구조를 제공한다. 사람이 읽기 쉬운 형태의 자료구조를 제공한다.

### 📌 Pandas 사용법

##### CSV 파일 불러오기 
```python
import pandas as pd

file_url = 'DATA_URL'
sample = pd.read_csv(file_url)
```

##### 데이터프레임을 살펴보는 다양한 함수
- 주요 자료형
	- `object, int64, float64, bool, datetime64`
- 통계적 정보
	- `count, mean, std, min, max, 25%, 50%, 75%` 
```python
# 앞부분의 데이터 출력
sample.head()
# 뒷부분의 데이터 출력
sample.tail()
# 데이터에 대한 요약 정보 출력
sample.info()
# 데이터에 대한 통계 정보 출력
sample.describe()
```

##### 데이터프레임 직접 만들기
```python
# 딕셔너리 형태로 만들기
sample_dic = {
			  'name' : ['John', 'Ann', 'Kevin'],
			  'age' : [23, 22, 21]
}
pd.DataFrame(sample_dic)

# columns 와 index 매개변수를 추가해 데이터프레임 만들기
sample_array = [
				['John', 23],
				['Ann', 22],
				['Kevin', 21]
]
# index 는 보통 0~n 으로 기본적으로 정의되기 때문에 정의하지 않는다.
pd.DataFrame(sample_array, columns = ['name', 'age'], index = ['a', 'b', 'c'])
```

##### 데이터프레임 인덱싱
- 데이터프레임의 특정 행과 열에서 데이터의 일부를 선택하는 것을 `인덱싱` 이라고 한다.
```python
# 컬럼 기준 인덱싱
sample_df['name']
sample_df[['name', 'age']]

# 행(index 이름) 기준 인덱싱
sample_df.loc['a']
sample_df.loc[['a', 'b', 'c']]
sample_df.loc['a':'c']

# 행(위치) 기준 인덱싱
sample_df.loc[0]
sample_df.loc[[0, 1, 2]]
sample_df.loc[0:2]

# 행과 열 동시 인덱싱
sample_df.loc[0:3, 2:4]
```

##### 컬럼 or 행 제거
- **drop()** 함수는 기본적으로 행을 제거하도록 설정되어 있다. 따라서, 컬럼을 제거할 때는 `axis=1` 옵션을 지정해주어야 한다.
```python
# 컬럼 제거
sample_df.drop('name', axis=1)
sample_df.drop(['name', 'age'], axis=1) # 두 개 이상 제거 시 리스트로 묶는다.

# 행 제거
sample_df.drop['a']
sample_df.drop['a', 'b']
```

##### 데이터프레임 인덱스 변경
- `reset_index` 와 `set_index` 로 인덱스를 특정 변수로 대체할 수도 있고, 인덱스를 별도의 변수로 빼내올 수 있다.
```python
# 인덱스 제거
sample_df.reset_index()
sample_df.reset_index(drop=True) # 기존 인덱스를 제거하되 새로운 컬럼을 추가하고 싶지 않다면

# 'name' 을 인덱스로 추가
sample_df.set_index('name')
```

##### 데이터프레임 기타 계산 함수들
```python
# 변수별 계산
sample_df.sum()  # 합
sample_df.mean() # 평균
sample_df.median() # 중간값
sample_df.var()  # 분산
sample_df.std()  # 표준편차
sample_df.count() # 데이터의 개수
sample_df.aggregate(['sum', 'mean'])  # 합과 평균 함께보기

# 그룹별 계산
iris.groupby('class').mean()  # class 별 평균
iris.groupby('class').agg(['count', 'mean'])  # class 별 데이터 개수와 평균

# 변수 내 고유값
iris['class'].unique()
iris['class'].nunique()
iris['class'].value_counts()
```

##### 데이터프레임 합치기
- 데이터를 결합하는 방법은 크게 **Inner Join, Full Join, Left Join, Right Join** 이 있다.
- 데이터는 **key** 값을 기준으로 합친다.
```python
# 내부 조인
left.merge(right)
left.merge(right, on = {'원하는 변수'})

# 전체 조인
left.merge(right, how = 'outer')

# 왼쪽 조인
left.merge(right, how = 'left')

# concat 함수로 결합
pd.concat([left, right])
pd.concat([left, right], axis=1) # 열 기준 결합
```
- **merge()** 함수는 기본적으로 내부 조인을 수행하며 특정 컬럼을 기준으로 데이터를 합친다.
- **join()** 함수는 인덱스를 기준으로 데이터를 합친다.
- **concat()** 함수는 기본적으로 행을 기준으로 합치며, `axis=1` 옵션을 사용하면 열 깆누으로 합칠 수 있다. left/right 조인은 지원하지 않고, inner/outer 조인만 가능하다.

## 🔥 Numpy

`Numpy` 는 컴퓨터가 계산하기 좋은 형태로 데이터를 제공한다. 그로 인해 빠른 연산 속도와 효율적인 메모리 사용으로 수치 계산 및 데이터 분석에 용이하다. `Pandas` 와의 차이점은 같은 자료형만 원소로 가질 수 있고, 행렬과 벡터 연산을 기반이며 3차원 이상의 배열도 가능하다는 점이다.

### 📌 Numpy 사용법

##### 기초 사용 예제
```python
import numpy as np

# 배열 생성
np.array([1, 2, 3])  # 1차원
np.array([           # 2차원
		[1, 2, 3],
		[4, 5, 6],
		[7, 8, 9]
])

# DataFrame을 ndarray로
sample_np = np.array(sample_df)

# ndarray를 DataFrame으로
pd.DataFrame(sample_np, columns = sample_df.columns)
```

##### 배열 인덱싱
```python
sample_np[0]          # 0행 출력
sampe_np[0, 2]        # 0행의 2번째 열 출력
sample_np[0:3, 2:4]   # 0~2행과 2~3열 출력
sample_np[:, 2]       # 모든 행의 2번째 열 출력
```

##### 배열의 연산
```python
# 배열 생성
np_a = np.array([[1, 3], [0, -2]])
np_b = np.array([[1, 0], [0, 1]])

# 산술 연산
np_a + 10
np_a - 5

# 배열과 배여ㅕㄹ의 연산
np_a + np_b
np_a - np_b
np_a * np_b  # 같은 자리 원소끼리의 곱셈
np_a @ np_b  # 행렬의 곱
```

##### 기타 함수들
```python
# 임의의 숫자 얻기
np.random.randint(11)
np.random.randint(50, 71)    # 지정된 범위 내에서
np.random.randint(50, 71, 5) # 지정된 범위 내에서 5개의 random 값

# 주어진 목록에서 임의의 값 선택
# 복원 추출
np.random.choice(['red', 'green', 'white', 'black', 'blue'], size=3)
# 비복원 추출
np.random.choice(['red', 'green', 'white', 'black', 'blue'], size=3, replace=False)

# 일련의 숫자 만들기
np.arange(1, 11)      # 1~10까지 배열 만들기
np.arange(1, 11, 2)   # 1~10까지 단, 2칸씩 건너뛰고
np.linspace(1, 10, 4) # 1~10까지 같은 간격으로 나눈 4개의 지점 찾기
```

## 🔥 요약
- **Pandas** 는 DataFrame 과 Series 를 자료구조로 제공한다. 각 자료구조에는 인덱스와 컬럼명이 있다.
	- DataFrame 은 여러 Series 를 합친 형태이다.
- **Numpy** 는 배열을 자료구조로 제공한다. 이로 인해 빠른 수치 계산이 가능하다.
	- Pandas와 다르게 인덱스와 컬럼명이 없다.