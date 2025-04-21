import pickle
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import re

# 출력 제한 해제
pd.set_option('display.max_colwidth', None)  # 한 셀에 표시할 최대 길이 제한 없음
pd.set_option('display.max_columns', None)   # 열 생략 없이 모두 표시
pd.set_option('display.expand_frame_repr', False)  # 줄바꿈 없이 한 줄에 표시

# --- CSV 불러오기 ---
df = pd.read_csv('./crawling_data/data.csv')  # ← 너 데이터에 맞게 변경
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.dropna(subset=['title', 'synopsis', 'genre']).reset_index(drop=True)
print(df.head())
df.info()

# --- 텍스트 & 장르 설정 ---

ALLOWED_GENRES = [
    'Reality TV', 'SF', '가족', '공포', '다큐멘터리',
    '드라마', '로맨스', '범죄', '스포츠', '액션', '역사', '코미디', '판타지'
]

def clean_genres(genre_str):
    genre = [g.strip() for g in genre_str.split(',') if g.strip() != '']
    return [g for g in genre if g in ALLOWED_GENRES]

X = df['synopsis'].fillna('')
Y = df['genre'].fillna('').apply(clean_genres)  # 필터링 적용

# ✅ 한글 없는 시놉시스 제거
has_korean = X.apply(lambda x: bool(re.search('[가-힣]', x)))
X = X[has_korean].reset_index(drop=True)
Y = Y[has_korean].reset_index(drop=True)

# --- 멀티라벨 인코더 불러오기 ---
with open('./models/encoder_multilabel.pickle', 'rb') as f:
    mlb = pickle.load(f)
labels = mlb.classes_
print("장르 클래스 목록:", labels)

y_true = mlb.transform(Y)  # 실제 라벨을 멀티-핫 형태로 변환

# --- 형태소 분석 & 정제 ---
okt = Okt()
for i in range(len(X)):
    X[i] = re.sub('[^가-힣]', ' ', X[i])
    X[i] = okt.morphs(X[i], stem=True)

for i in range(len(X)):
    X[i] = ' '.join([word for word in X[i] if len(word) > 1])

# --- 토크나이저 불러오기 ---
with open('./models/token_max_250.pickle', 'rb') as f:
    token = pickle.load(f)

tokened_x = token.texts_to_sequences(X)

# --- 시퀀스 자르기 및 패딩 (max_len = 25 기준) ---
for i in range(len(tokened_x)):
    tokened_x[i] = tokened_x[i][:25]
x_pad = pad_sequences(tokened_x, maxlen=25)

# ✅ 정답이 비어있는 샘플 제거
non_empty_indices = [i for i, g in enumerate(Y) if len(g) > 0]
df = df.iloc[non_empty_indices].reset_index(drop=True)
Y = [Y[i] for i in non_empty_indices]
y_true = y_true[non_empty_indices]
x_pad = x_pad[non_empty_indices]

# --- 모델 불러오기 ---
model = load_model('./models/multilabel_classification_model_0.4815.h5')  # 너 모델 경로에 맞게 수정

# --- 예측 ---
y_pred = model.predict(x_pad)
print("예측 완료:", y_pred.shape)

# --- 예측 라벨 추출 (실제 장르 개수만큼 확률 높은 것 선택) ---
predict_section = []
for i in range(len(y_pred)):
    n_labels = int(np.sum(y_true[i]))
    top_n = y_pred[i].argsort()[-n_labels:][::-1]
    pred_label = [labels[j] for j in top_n]
    predict_section.append(pred_label)

df['predict'] = predict_section
df['genre'] = Y
print(df[[ 'title', 'genre', 'predict']].head(50))

# --- 평가 준비 ---
df['predict'] = predict_section
df['OX_strict'] = 0  # 완전 정답률 (예측 == 정답)
df['OX_loose'] = 0   # 부분 정답률 (예측 ∩ 정답 ≥ 1개)

for i in range(len(df)):
    true_set = set(Y[i])
    pred_set = set(df.loc[i, 'predict'])

    if true_set == pred_set:  # 완전 일치
        df.loc[i, 'OX_strict'] = 1

    if len(true_set & pred_set) > 0:  # 교집합 1개 이상
        df.loc[i, 'OX_loose'] = 1

# --- ① 완전 정답률 ---
strict_acc = df['OX_strict'].mean()
print("✅ 완전 정답률 (정확히 일치):", strict_acc)

# --- ② 부분 정답률 ---
loose_acc = df['OX_loose'].mean()
print("✅ 부분 정답률 (하나 이상 일치):", loose_acc)