# 07. 나이브 베이즈 : 스팸 여부 판단하기

## 🔥 Naive Bayes 란?

**Naive Bayes** 란, 베이즈 정리를 적용한 조건부 확률 기반의 분류 모델이다.
최근 딥러닝이 자연어 처리에 더 탁월한 모습을 보여주지만, 딥러닝보다 간단한 방법으로 자연어 처리를 하고자 할 때 좋은 선택이 될 수 있다.

- 장점
	- 비교적 간단한 알고리즘에 속하며 속도 또한 빠르다.
	- 작은 훈련셋으로도 잘 예측한다.
- 단점
	- 모든 독립변수가 각각 독립적임을 전제로 한다.
	- 이는 장점이 되기도 하고 단점이 되기도 한다.
	- 모든 독립변수들이 모두 독립적이라면, 우수한 알고리즘일 수 있지만 실제 데이터에서 그런 경우가 많지 않은 것이 단점이 된다.

## 🔥 문자 데이터셋을 이용한 스팸 메일 여부 판단

### 1. 데이터 불러오기

```python
import pandas as pd

file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/spam.csv'  
data = pd.read_csv(file_url)
```

### 2. 데이터셋 확인

```python
print(data.head())
print(data['target'].unique())
```

데이터셋은 독립변수 1개와 목표변수 1개로 구성되어 있다.
- 독립변수는 `text` , 목표변수는 `ham, spam (스팸문자와 아닌 문자)` 으로 구성되어있다.

### 3. 전처리 : 특수 기호 제거

자연어를 다룰 때 데이터의 기준은 **단어** 이다.
쉼표, 마침표와 같은 특수 기호는 단어를 처리할 때 노이즈가 되므로 제거해야한다.

```python
import string  
print(string.punctuation)
```

`string` 라이브러리의 **punctuation** 을 실행하면 특수 기호 목록을 확인할 수 있다.
이를 이용해 특수 기호를 제거하는 메소드를 아래와 같이 만든다.

```python
def remove_punc(x):  
    new_string = []  
    for i in x:  
        if i not in string.punctuation:  
            new_string.append(i)  
    new_string = ''.join(new_string)  
    return new_string  
```

 - 인자로 들어오는 문자열을 한 단어씩 가지고와 특수 기호가 아닌 문자들만 `new_string` 에 추가한다.
 - 최종적으로 다시 문자열로 만들어 반환한다.

이것을 데이터셋 `text` 전체에 적용하면 모든 문자들이 하나의 문자열로 반환되어 저장된다.
데이터 프레임의 시리즈 한 줄을 따로 함수에 적용시키기 위해서 **apply()** 메소드를 사용할 수 있다.

```python
# 데이터셋 업데이트 : 특수 기호 제거  
data['text'] = data['text'].apply(remove_punc)
```

### 4. 전처리 : 불용어 처리

**불용어 (stopword)** 는 자연어 분석에 큰 도움이 안되는 단어를 의미한다.
이러한 단어를 제거하면 데이터를 조금이나마 가볍게 만들 수 있다.

```python
import nltk  
nltk.download('stopwords')  
  
from nltk.corpus import stopwords  
print(stopwords.words('english'))
```

`nltk` 라이브러리에서 불용어 목록을 다운로드 받아 import 하면, 영어 불용어 목록을 확인할 수 있다.
이를 통해 불용어를 제거해주는 메소드를 아래와 같이 만든다.

```python
def stop_words(x):  
    new_string = []  
    for i in x.split():  
        if i.lower() not in stopwords.words('english'):  
            new_string.append(i.lower())  
    new_string = ' '.join(new_string)  
    return new_string
```

- 인자로 들어오는 문자열을 단어 단위로 구분한 후, 해당 단어가 불용어가 아니라면 `new_string` 에 추가한다.
- 최종적으로 다시 문자열로 만들어 반환한다.

```python
data['text'] = data['text'].apply(stop_words)
```

특수 기호를 제거한 것과 같이 **apply()** 메소드를 이용해 데이터셋에 적용한다.

### 5. 전처리 : 목표 컬럼 형태 변경

현재 `target` 은 `ham 과 spam` 으로 이루어져있다.
문자 형식의 데이터도 모델링에 에러를 유발하지는 않지만, 때에 따라 해석에 문제가 생길 수도 있기 때문에 숫자로 변환한다.
- 스팸은 1, 스팸이 아닌 문자는 0으로 변환한다.

```python
data['target'] = data['target'].map({'spam': 1, 'ham': 0})
```

### 6. 전처리 : 카운트 기반으로 벡터화

카운트 기반 벡터화는 문자를 개수 기반으로 벡터화하는 방식이다.
데이터 전체에 존재하는 모든 단어들을 사전처럼 모은 후, 인덱스를 부여하고 문장마다 속한 단어가 있는 인덱스를 카운트한다.

```python
x = data['text']  
y = data['target']  
  
from sklearn.feature_extraction.text import CountVectorizer  
  
cv = CountVectorizer()  
cv.fit(x)  
x = cv.transform(x)
```

- `CountVectorizer` 를 학습하고, **transform()** 메소드로 변환한다.

### 7. 모델링 및 예측/평가

모델을 학습하고, 결과를 평가한다.

```python
from sklearn.model_selection import train_test_split  

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)  
  
# 모델 import
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()  
model.fit(x_train, y_train)  
pred = model.predict(x_test)  
  
# 평가  
from sklearn.metrics import accuracy_score, confusion_matrix  
  
print(accuracy_score(y_test, pred))     # 정확도  
print(confusion_matrix(y_test, pred))   # 혼동 행렬  
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='.0f')  
plt.show()
```

정확도는 **0.9856502242152466** 로, 약 **98.9%** 이며 높은 예측률을 보인다.

여기서 **혼동 행렬 (confusion matrix)** 모듈도 사용해 결과를 평가한다.
- 혼동 행렬은 실제값과 예측값이 어떻게 분포되었는지를 행렬로 나타낸다.

혼동 행렬 출력 결과는 다음과 같다.
- 스팸이 아닌데, 스팸이 아니라고 예측한 경우 : 965건
- 스팸이 아닌데, 스팸이라고 예측한 경우 : 12건
- 스팸인데, 스팸이 아니라고 예측한 경우 : 4건
- 스팸인데, 스팸이라고 예측한 경우 : 134건

여기서 12건에 해당하는 것을 **1종 오류** , 4건에 해당하는 것을 **2종 오류** 라고 한다.
두 오류는 성격이 다른 오류이며, 때에 따라 둘 중 한쪽이 더 중요하다.
- 1종 오류는 스팸이 아닌데 스팸으로 예측해 중요한 문자를 필터링할 수도 있다.
- 2종 오류는 스팸을 받게는 되겠지만, 중요한 문자를 필터링 하지 않는다.
