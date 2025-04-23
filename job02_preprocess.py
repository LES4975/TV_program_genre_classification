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
import matplotlib.pyplot as plt     # ✅ 추가
import seaborn as sns               # ✅ 추가
from collections import Counter     # ✅ 추가
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')  # 윈도우일 경우
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

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
titles = df['title']  # ✅ 추가

# --- X, Y 합쳐서 DataFrame 만들기 (오버샘플링 함수는 df 필요) ---
df_xy = pd.DataFrame({'title': titles, 'synopsis': X, 'genre': Y})  # ✅ 'title' 포함

# 🔸 오버샘플링 전 카운트 계산
before_counts = Counter([g for genres in df_xy['genre'] for g in genres])

# ✅ 오버샘플링 적용 (개별 장르 기준)
df_xy = oversample_by_individual_label(df_xy, ALLOWED_GENRES)

# 🔸 오버샘플링 후 카운트 계산
after_counts = Counter([g for genres in df_xy['genre'] for g in genres])

# 🔸 정렬 기준: 오버샘플링 후 수 기준 내림차순
sorted_genres = sorted(after_counts, key=after_counts.get, reverse=True)

# 🔸 히트맵용 데이터프레임 구성
heatmap_df = pd.DataFrame({
    '오버샘플링 전': [before_counts[g] for g in sorted_genres],
    '오버샘플링 후': [after_counts[g] for g in sorted_genres]
}, index=sorted_genres)

# 🔸 히트맵 그리기
# 🔸 히트맵 그리기 (흑백 스타일)
plt.figure(figsize=(6, 8))
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt="d",
    cmap="Greys",          # ⚫ 흑백 계열 컬러맵
    linewidths=1,          # 🔲 셀 경계 강조
    linecolor='black',
    cbar=False             # 컬러 바 제거
)
plt.title("장르별 데이터 분포 (오버샘플링 전 vs 후)")
plt.xlabel("단계")
plt.ylabel("장르")
plt.tight_layout()

# 🔸 이미지 저장
plt.savefig("oversampling_heatmap.png")
plt.show()

# --- 📊 수치 포함된 막대 겹침 그래프 ---
before = [before_counts[g] for g in sorted_genres]
after = [after_counts[g] for g in sorted_genres]
x = np.arange(len(sorted_genres))
bar_width = 0.6

plt.figure(figsize=(10, 6))
# 회색: 오버샘플링 후
bars2 = plt.bar(x, after, width=bar_width, color='lightgray', label='오버샘플링 후')
# 검은색: 오버샘플링 전
bars1 = plt.bar(x, before, width=bar_width, color='black', label='오버샘플링 전')



# 수치 추가
for i in range(len(x)):
    # 오버샘플링 전 (검정 위)
    plt.text(x[i], before[i] + 500, f'{before[i]:,}', color='black', ha='center', va='bottom', fontsize=8)
    # 오버샘플링 후 (회색 위)
    plt.text(x[i], after[i] + 500, f'{after[i]:,}', color='black', ha='center', va='bottom', fontsize=8)

plt.xticks(x, sorted_genres, rotation=45, ha='right')
plt.ylabel('샘플 수')
plt.title('장르별 데이터 분포 (오버샘플링 전 vs 후)', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig("oversampling_bar_comparison_annotated.png", dpi=300)
plt.show()


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

df_xy = pd.DataFrame({'title': df_xy['title'], 'synopsis': X, 'genre': Y})
df_xy = df_xy[
    (df_xy['title'].astype(str).str.strip() != '') &
    (df_xy['synopsis'].astype(str).str.strip() != '') &
    (df_xy['genre'].apply(lambda g: isinstance(g, list) and len(g) > 0))
].reset_index(drop=True)

# 다시 X, Y, x_pad, y 재정의
X = df_xy['synopsis']
Y = df_xy['genre']
x_pad = x_pad[df_xy.index]
multi_hot_y = mlb.transform(Y)

# 8. 학습/테스트 분리
x_train, x_test, y_train, y_test = train_test_split(x_pad, multi_hot_y, test_size=0.1, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 9. 저장
np.save('./crawling_data/title_x_train_wordsize{}'.format(wordsize), x_train)
np.save('./crawling_data/title_x_test_wordsize{}'.format(wordsize), x_test)
np.save('./crawling_data/title_y_train_wordsize{}'.format(wordsize), y_train)
np.save('./crawling_data/title_y_test_wordsize{}'.format(wordsize), y_test)

