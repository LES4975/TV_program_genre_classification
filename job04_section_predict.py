import pickle
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
import re
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('default')
matplotlib.rc('font', family='Malgun Gothic')  # 윈도우일 경우
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지


# 출력 제한 해제
pd.set_option('display.max_colwidth', None)  # 한 셀에 표시할 최대 길이 제한 없음
pd.set_option('display.max_columns', None)   # 열 생략 없이 모두 표시
pd.set_option('display.expand_frame_repr', False)  # 줄바꿈 없이 한 줄에 표시

# --- CSV 불러오기 ---
df = pd.read_csv('./crawling_data/justwatch_test_2025.csv')  # ← 너 데이터에 맞게 변경
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
df = df[has_korean].reset_index(drop=True)  # ✅ 추가

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

# ✅ 시놉시스가 비어있는 행 제거 (정제 후 기준)
non_empty_indices = [i for i, text in enumerate(X) if text.strip() != '']
X = [X[i] for i in non_empty_indices]
Y = [Y[i] for i in non_empty_indices]
df = df.iloc[non_empty_indices].reset_index(drop=True)
y_true = y_true[non_empty_indices]

# --- 토크나이저 불러오기 ---
with open('./models/token_max_273.pickle', 'rb') as f:
    token = pickle.load(f)

tokened_x = token.texts_to_sequences(X)

# --- 시퀀스 자르기 및 패딩 (max_len = 25 기준) ---
for i in range(len(tokened_x)):
    tokened_x[i] = tokened_x[i][:273]
x_pad = pad_sequences(tokened_x, maxlen=273)

# ✅ 정답이 비어있는 샘플 제거
non_empty_indices = [i for i, g in enumerate(Y) if len(g) > 0]
df = df.iloc[non_empty_indices].reset_index(drop=True)
Y = [Y[i] for i in non_empty_indices]
y_true = y_true[non_empty_indices]
x_pad = x_pad[non_empty_indices]

# --- 모델 불러오기 ---
model = load_model('./models/multilabel_classification_model_0.5283.h5')  # 너 모델 경로에 맞게 수정

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
print(df[['title', 'genre', 'predict']].head(50))

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

def partial_match_accuracy(y_true_bin, y_pred_bin):
    scores = []
    for yt, yp in zip(y_true_bin, y_pred_bin):
        true_positives = np.sum(np.logical_and(yt, yp))
        actual_positives = np.sum(yt)
        score = true_positives / actual_positives if actual_positives > 0 else 1
        scores.append(score)
    return np.mean(scores)

# ---  완전 정답률 ---
strict_acc = df['OX_strict'].mean()
print("✅ 완전 정답률 (정확히 일치):", strict_acc)

# ---  부분 정답률 ---
loose_acc = df['OX_loose'].mean()
print("✅ 부분 정답률 (하나 이상 일치):", loose_acc)

# ---  샘플 단위 부분 정답률
y_pred_bin = (y_pred > 0.5).astype(int)
partial_acc = partial_match_accuracy(y_true, y_pred_bin)
print("✅ 정답 대비 예측률 (정답 중 몇 개 맞췄는지): {:.5f}".format(partial_acc))

# --- 📊 평가 지표 시각화 ---
# 평가 결과
labels = ['완전 정답률', '부분 정답률', '정답 대비 예측률']
scores = [strict_acc, loose_acc, partial_acc]

fig, ax = plt.subplots(figsize=(6, 5))
bars = []

x = range(len(labels))  # 0, 1, 2

# 막대그래프 (세로 방향)
for i, score in enumerate(scores):
    if i == 0:
        bar = ax.bar(i, score, color='white', edgecolor='black', width=0.4)
    elif i == 1:
        bar = ax.bar(i, score, color='white', edgecolor='black', hatch='//', width=0.4)
    else:
        bar = ax.bar(i, score, color='black', width=0.4)
    bars.append(bar)

# 축 설정
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_ylabel('정확도', fontsize=12)
ax.set_title('장르 분류 모델 평가 지표', fontsize=14, fontweight='bold')
ax.grid(axis='y', linestyle='--', linewidth=0.5)

# 수치 표시
for i, score in enumerate(scores):
    ax.text(i, score + 0.02, f'{score*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("accuracy_vertical_bar.png", dpi=300)
plt.show()

# --- 📊 장르별 예측 정확도 계산 및 시각화 ---
genre_accuracies = {}
true_labels = mlb.classes_  # ✅ 실제 장르 이름

for idx, genre in enumerate(true_labels):  # ✅ 이걸로 수정
    true_pos = np.sum((y_true[:, idx] == 1) & (y_pred_bin[:, idx] == 1))
    total = np.sum(y_true[:, idx] == 1)
    acc = true_pos / total if total > 0 else 0
    genre_accuracies[genre] = acc

# ✅ 정확도 높은 순으로 정렬하되, 가장 높은 걸 위로 보이게 역순 전달
sorted_items = sorted(genre_accuracies.items(), key=lambda x: x[1], reverse=True)
genres, accs = zip(*sorted_items)

# ✅ 역순으로 넘김
genres = genres[::-1]
accs = accs[::-1]

# ✅ 색상: 진한 회색부터 연한 회색까지
from matplotlib import cm
colors = cm.Greys(np.linspace(0.4, 0.85, len(accs)))

# 시각화
plt.figure(figsize=(8, 5))
bars = plt.barh(genres, accs, color=colors)

# ✅ 수치 표시 (퍼센트)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{acc * 100:.0f}%', va='center', fontsize=9)

plt.title('장르별 예측 정확도', fontsize=14, fontweight='bold')
plt.xlabel('정확도', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("genre_accuracy_chart_gradient.png", dpi=300)
plt.show()