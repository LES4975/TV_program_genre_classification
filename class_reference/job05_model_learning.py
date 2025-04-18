import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *

x_train = np.load('./crawling_data/title_x_train_wordsize12016.npy', allow_pickle=True)
x_test = np.load('./crawling_data/title_x_test_wordsize12016.npy', allow_pickle=True)
y_train = np.load('./crawling_data/title_y_train_wordsize12016.npy', allow_pickle=True)
y_test = np.load('./crawling_data/title_y_test_wordsize12016.npy', allow_pickle=True)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

model = Sequential()
model.add(Embedding(12016, 300))
model.build(input_shape=(None, 20))
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(46, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
fit_hist = model.fit(x_train, y_train, batch_size=128,
                     epochs=10, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Final test set accuracy', score[1])
model.save('./models/news_section_classfication_model_{}.h5'.format(score[1]))
plt.plot(fit_hist.history['val_accuracy'], label='val accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()
