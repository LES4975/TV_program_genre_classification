import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPool1D, LSTM, Dropout, Flatten, Dense

# --- 저장된 전처리 결과 불러오기 ---
x_train = np.load('./crawling_data/title_x_train_wordsize5165.npy', allow_pickle=True)
x_test = np.load('./crawling_data/title_x_test_wordsize5165.npy', allow_pickle=True)
y_train = np.load('./crawling_data/title_y_train_wordsize5165.npy', allow_pickle=True)
y_test = np.load('./crawling_data/title_y_test_wordsize5165.npy', allow_pickle=True)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# --- 멀티라벨 모델 구성 ---
model = Sequential()
model.add(Embedding(5165, 300))  # 단어 사전 크기와 임베딩 차원은 전처리에서 확인한 값
model.build(input_shape=(None, x_train.shape[1]))  # 시퀀스 길이 자동 설정
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(16, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(16, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))  # ← softmax → sigmoid
model.summary()

# --- 멀티라벨용 컴파일 ---
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- 학습 시작 ---
fit_hist = model.fit(x_train, y_train,
                     batch_size=128,
                     epochs=30,
                     validation_data=(x_test, y_test))

# --- 평가 및 저장 ---
score = model.evaluate(x_test, y_test, verbose=0)
print('Final test set accuracy:', score[1])

model.save('./models/multilabel_classification_model_{:.4f}.h5'.format(score[1]))

# --- 시각화 ---
plt.plot(fit_hist.history['val_accuracy'], label='val accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()