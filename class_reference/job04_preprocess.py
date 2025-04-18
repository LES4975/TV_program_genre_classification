import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt, Komoran
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re

df = pd.read_csv('./crawling_data/news_titles.csv')
df.info()
print(df.head())
print(df.category.value_counts())

X = df.titles
Y = df.category

# print(X[0])
okt = Okt()
# okt_x = okt.morphs(X[0])
# print(okt_x)
# okt_x = okt.morphs(X[0], stem=True)
# print(okt_x)
#
# komoran = Komoran()
# komoran_x = komoran.morphs(X[0])
# print(komoran_x)

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:5])
label = encoder.classes_
print(label)
with open('./models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

onehot_y = to_categorical(labeled_y)
print(onehot_y)

# cleand_x = re.sub('[^가-힣]',' ', X[0])
# print(X[0])
# print(cleand_x)

for i in range(len(X)):
    X[i] = re.sub('[^가-힣]', ' ', X[i])
    X[i] = okt.morphs(X[i], stem=True)
    if i % 1000 == 0:
        print(i)
print(X)
for idx, sentence in enumerate(X):
    # print(sentence)
    words = []
    for word in sentence:
        # print(word)
        if len(word) > 1:
            words.append(word)
    # print(words)
    X[idx] = ' '.join(words)
    # print(X[idx])

print(X[:10])

token = Tokenizer()
token.fit_on_texts(X)
tokened_x = token.texts_to_sequences(X)
print(tokened_x)
wordsize = len(token.word_index) + 1


max = 0

for sentence in tokened_x:
    if max < len(sentence):
        max = len(sentence)

print(max)
with open('./models/token_max_{}.pickle'.format(max), 'wb') as f:
    pickle.dump(token, f)

x_pad = pad_sequences(tokened_x, max)
print(x_pad)

x_train, x_test, y_train, y_test = train_test_split(
    x_pad, onehot_y, test_size=0.1)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

np.save('./crawling_data/title_x_train_wordsize{}'.format(wordsize),x_train)
np.save('./crawling_data/title_x_test_wordsize{}'.format(wordsize),x_test)
np.save('./crawling_data/title_y_train_wordsize{}'.format(wordsize),y_train)
np.save('./crawling_data/title_y_test_wordsize{}'.format(wordsize),y_test)

