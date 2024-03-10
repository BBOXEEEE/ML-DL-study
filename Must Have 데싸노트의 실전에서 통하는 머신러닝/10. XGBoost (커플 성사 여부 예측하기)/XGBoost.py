import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 불러오기
file_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/dating.csv'
data = pd.read_csv(file_url)

# 데이터셋 확인
print(data.head())
print(data.info())
print(round(data.describe(), 2))

print('#' * 50, 1)
print()

# 전처리 : 결측치 처리
print(data.isna().mean())
# 중요도 관련 변수는 결측치 제거
data = data.dropna(subset=['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny',
                           'pref_o_ambitious', 'pref_o_shared_interests', 'attractive_important',
                           'sincere_important', 'intellicence_important', 'funny_important',
                           'ambtition_important', 'shared_interests_important'])
# 그 외의 결측치는 -99 (응답하지 않음) 으로 채움
data = data.fillna(-99)

print('#' * 50, 2)
print()


# 전처리 : 피처 엔지니어링
# (1) 나이와 관련된 변수
def age_gap(x):
    if x['age'] == -99:
        return -99
    elif x['age_o'] == -99:
        return -99
    elif x['gender'] == 'female':
        return x['age_o'] - x['age']
    else:
        return x['age'] - x['age_o']


data['age_gap'] = data.apply(age_gap, axis=1)   # age_gap 변수에 age_gap 함수 적용
data['age_gap_abs'] = abs(data['age_gap'])  # 절대값 적용


# (2) 인종과 관련된 변수
def same_race(x):
    if x['race'] == -99:
        return -99
    elif x['race_o'] == -99:
        return -99
    elif x['race'] == x['race_o']:
        return 1
    else:
        return -1


data['same_race'] = data.apply(same_race, axis=1)   # same_race 함수 적용


def same_race_point(x):
    if x['same_race'] == -99:
        return -99
    else:
        return x['same_race'] * x['importance_same_race']


data['same_race_point'] = data.apply(same_race_point, axis=1)


# (3) 평가/중요도 변수
def rating(x, importance, score):
    if x[importance] == -99:
        return -99
    elif x[score] == -99:
        return -99
    else:
        return x[importance] * x[score]


partner_imp = data.columns[8:14]    # 상대방의 중요도
partner_rate_me = data.columns[14:20]   # 본인에 대한 상대방의 평가
my_imp = data.columns[20:26]    # 본인의 중요도
my_rate_partner = data.columns[26:32]   # 상대방에 대한 본인의 평가

new_label_partner = ['attractive_p', 'sincere_partner_p', 'intelligence_p',
                     'funny_p', 'ambition_p', 'shared_interests_p']
new_label_me = ['attractive_m', 'sincere_partner_m', 'intelligence_m',
                'funny_m', 'ambition_m', 'shared_interests_m']
# rating 함수 적용
for i, j, k in zip(new_label_partner, partner_imp, partner_rate_me):
    data[i] = data.apply(lambda x: rating(x, j, k), axis=1)

for i, j, k in zip(new_label_me, my_imp, my_rate_partner):
    data[i] = data.apply(lambda x: rating(x, j, k), axis=1)

# (3) 더미 변수로 변환
data = pd.get_dummies(data, columns=['gender', 'race', 'race_o'], drop_first=True)

print('#' * 50, 3)
print()

# 모델링 및 평가
# (1) 데이터 분할
from sklearn.model_selection import train_test_split

X = data.drop('match', axis=1)
y = data['match']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# (2) 모델링
import xgboost as xgb

model = xgb.XGBClassifier(n_estimators=500, max_depth=5, random_state=100)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# 평가
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(accuracy_score(y_test, pred))     # 정확도
print(confusion_matrix(y_test, pred))   # 혼동 행렬
print(classification_report(y_test, pred))  # 오류 유형에 따른 평가

print('#' * 50, 4)
print()

# 하이퍼파라미터 튜닝 : 그리드 서치
from sklearn.model_selection import GridSearchCV

parameters = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [5, 7, 10],
    'subsample': [0.5, 0.7, 1],
    'n_estimators': [300, 500, 1000]
}

# (1) 그리드 서치 적용
model = xgb.XGBClassifier()
gs_model = GridSearchCV(model, parameters, n_jobs=1, scoring='f1', cv=5)
gs_model.fit(X_train, y_train)
print(gs_model.best_params_)

# (2) 평가
pred = gs_model.predict(X_test)
print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

print('#' * 50, 5)
print()

# 중요 변수 확인
model = xgb.XGBClassifier(learning_rate=0.3, max_depth=5, n_estimators=1000, subsample=1, random_state=100)
model.fit(X_train, y_train)
print(model.feature_importances_)

# 데이터 프레임으로 변경
feature_imp = pd.DataFrame({'features': X_train.columns, 'values': model.feature_importances_})
print(feature_imp.head())

# 그래프로 출력
plt.figure(figsize=(20, 10))
sns.barplot(x='values', y='features', data=feature_imp.sort_values(by='values', ascending=False).head(10))
plt.show()

