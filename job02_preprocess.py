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

# 1. 데이터 불러오기
df = pd.read_csv('./crawling_data/justwatch_test.csv')
# 혹시 모를 중복 행 제거 및 인덱스 재정렬
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

#장르에 쓸데없는 값은 안받기
def clean_genres(genre_str):
    genres = [g.strip() for g in genre_str.split(',') if g.strip() != '']
    filtered = []
    for g in genres:
        # 숫자 괄호 형태 제거: (123)
        if re.fullmatch(r'\(\d+\)', g):
            continue
        # 명백한 이상값 제거 (예: 권대현)
        if g == '권대현':
            continue
        filtered.append(g)
    return filtered

# 2. 텍스트 & 장르 설정
X = df['synopsis'].fillna('')
Y = df['genre'].fillna('').apply(clean_genres)


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

