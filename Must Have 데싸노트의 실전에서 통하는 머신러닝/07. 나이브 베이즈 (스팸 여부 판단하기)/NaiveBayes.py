import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/spam.csv'
data = pd.read_csv(file_url)

# 데이터셋 확인
print(data.head())
print(data['target'].unique())

print('#' * 50, 1)

# 전처리 : 특수 기호 제거
import string
print(string.punctuation)  # 특수 기호 목록


# 특수 기호를 제외한 새로운 문자열을 만들어주는 함수
def remove_punc(x):
    new_string = []
    for i in x:
        if i not in string.punctuation:
            new_string.append(i)
    new_string = ''.join(new_string)
    return new_string


# 테스트
sample_string = data['text'].loc[0]
print(remove_punc(sample_string))   # 데이터셋 중 하나의 문장
# print(remove_punc(data['text']))    # 데이터셋 전체
# print(data['text'].apply(remove_punc))  # 데이터프레임 한줄씩 함수 적용

# 데이터셋 업데이트 : 특수 기호 제거
data['text'] = data['text'].apply(remove_punc)

print('#' * 50, 2)

# 전처리 : 불용어 제거
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
# print(stopwords.words('english'))   # 영어 불용어 목록


# 불용어 제거 함수
def stop_words(x):
    new_string = []
    for i in x.split():
        if i.lower() not in stopwords.words('english'):
            new_string.append(i.lower())
    new_string = ' '.join(new_string)
    return new_string


# 테스트
sample_string = data['text'].loc[0]
print(stop_words(sample_string))

# 데이터셋 업데이트
data['text'] = data['text'].apply(stop_words)

# 전처리 : 목표 컬럼 형태 변경
data['target'] = data['target'].map({'spam': 1, 'ham': 0})

# 전처리 : 카운트 기반으로 벡터화
x = data['text']
y = data['target']

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv.fit(x)
x = cv.transform(x)

print('#' * 50, 3)

# 모델링 및 예측/평가
from sklearn.model_selection import train_test_split

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
