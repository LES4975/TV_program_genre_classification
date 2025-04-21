import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt, Komoran
from sklearn.preprocessing import MultiLabelBinarizer #멀티로 수정
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from sklearn.utils import resample  # ✅ 추가

ALLOWED_GENRES = [
    'Reality TV', 'SF', '가족', '공포', '다큐멘터리',
    '드라마', '로맨스', '범죄', '스포츠', '액션', '역사', '코미디', '판타지'
]
# ✅ 오버샘플링 함수 정의
from collections import Counter

def oversample_by_individual_label(df, allowed_genres, genre_col='genre'):
    # 1. 장르별 등장 횟수 세기
    all_labels = [g for sublist in df[genre_col] for g in sublist]
    label_counts = Counter(all_labels)

    # 2. 목표 등장 횟수 (가장 많은 장르 수로 맞춤)
    target_count = max(label_counts[g] for g in allowed_genres)

    # 3. 장르별로 부족한 만큼 복제
    dfs = [df]  # 원본 포함
    for genre in allowed_genres:
        genre_df = df[df[genre_col].apply(lambda x: genre in x)]
        count = label_counts[genre]
        if count < target_count:
            needed = target_count - count
            sampled = resample(genre_df, replace=True, n_samples=needed, random_state=42)
            dfs.append(sampled)

    df_balanced = pd.concat(dfs).reset_index(drop=True)
    return df_balanced

# 1. 데이터 불러오기
df = pd.read_csv('./crawling_data/data.csv')
# 혹시 모를 중복 행 제거 및 인덱스 재정렬
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df = df.dropna(subset=['title', 'synopsis', 'genre']).reset_index(drop=True)

def clean_genres(genres):
    if isinstance(genres, str):  # 혹시 문자열이면 split
        genres = [g.strip() for g in genres.split(',') if g.strip() != '']
    return [g for g in genres if g in ALLOWED_GENRES]


# --- 장르 전처리 ---
df['genre'] = df['genre'].fillna('').apply(clean_genres)
df['synopsis'] = df['synopsis'].fillna('')

# 2. 텍스트 & 장르 설정
X = df['synopsis']
Y = df['genre']

# --- X, Y 합쳐서 DataFrame 만들기 (오버샘플링 함수는 df 필요) ---
df_xy = pd.DataFrame({'synopsis': X, 'genre': Y})

# ✅ 오버샘플링 적용 (개별 장르 기준)
df_xy = oversample_by_individual_label(df_xy, ALLOWED_GENRES)

# --- 다시 분리 ---
X = df_xy['synopsis']
Y = df_xy['genre']

has_korean = X.apply(lambda x: bool(re.search('[가-힣]', x)))
X = X[has_korean].reset_index(drop=True)
Y = Y[has_korean].reset_index(drop=True)

# 3. 멀티 라벨 인코딩
mlb = MultiLabelBinarizer()
multi_hot_y = mlb.fit_transform(Y)

# 저장
with open('./models/encoder_multilabel.pickle', 'wb') as f:
    pickle.dump(mlb, f)

print(multi_hot_y[:5])
print("전체 장르 목록:", mlb.classes_)

# 4. 형태소 분석기 준비
okt = Okt()
komoran = Komoran()

# 5. 텍스트 정제 및 형태소 분석
for i in range(len(X)):
    X[i] = re.sub('[^가-힣]', ' ', X[i])
    X[i] = okt.morphs(X[i], stem=True)
    if i % 500 == 0:
        print(f"{i}번째 처리 중...")


# 6. 불용어 제거
for idx, sentence in enumerate(X):
    words = []
    for word in sentence:
        if len(word) > 1:
            words.append(word)
    X[idx] = ' '.join(words)

print("전처리 후 일부 문장:")
print(X[:5])

# 7. 토크나이저
token = Tokenizer()
token.fit_on_texts(X)
tokened_x = token.texts_to_sequences(X)
print(tokened_x[:3])

# 단어 수
wordsize = len(token.word_index) + 1
print("단어 사전 크기:", wordsize)

# 최대 길이
max_len = 0
for sentence in tokened_x:
    if max_len < len(sentence):
        max_len = len(sentence)
print("최대 시퀀스 길이:", max_len)

# 토크나이저 저장
with open('./models/token_max_{}.pickle'.format(max_len), 'wb') as f:
    pickle.dump(token, f)

# 시퀀스 패딩
x_pad = pad_sequences(tokened_x, maxlen=max_len)
print("패딩 결과:", x_pad.shape)

# 8. 학습/테스트 분리
x_train, x_test, y_train, y_test = train_test_split(x_pad, multi_hot_y, test_size=0.1, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 9. 저장
np.save('./crawling_data/title_x_train_wordsize{}'.format(wordsize), x_train)
np.save('./crawling_data/title_x_test_wordsize{}'.format(wordsize), x_test)
np.save('./crawling_data/title_y_train_wordsize{}'.format(wordsize), y_train)
np.save('./crawling_data/title_y_test_wordsize{}'.format(wordsize), y_test)

