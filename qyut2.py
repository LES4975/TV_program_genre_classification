import sys
import os
import re
import numpy as np
import pandas as pd
import google.generativeai as genai
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QMessageBox
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
from konlpy.tag import Okt
import matplotlib.pyplot as plt
from matplotlib import rc

# 📌 한글 폰트 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ✅ Gemini API 키 설정
genai.configure(api_key="AIzaSyD4Bslbe_qnWAUoY7OPWzNiBB7n8Kwom7I")  # 여기에 발급받은 키 넣기
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

# ✅ Keras 모델 및 전처리 객체 불러오기
model = load_model('./models/multilabel_classification_model_0.5342.h5')
with open('./models/token_max_273.pickle', 'rb') as f:
    token = pickle.load(f)
with open('./models/encoder_multilabel.pickle', 'rb') as f:
    mlb = pickle.load(f)

labels = mlb.classes_
okt = Okt()

# ✅ 시놉시스 전처리 함수
def preprocess_text(text):
    text = re.sub('[^가-힣]', ' ', text)
    words = okt.morphs(text, stem=True)
    words = [w for w in words if len(w) > 1]
    joined = ' '.join(words)
    seq = token.texts_to_sequences([joined])
    pad = pad_sequences(seq, maxlen=273)
    return pad

# ✅ Gemini 시놉시스 생성 함수
def generate_random_synopsis():
    prompt = "1.Reality TV, 2.SF, 3.가족, 4.공포, 5.다큐멘터리, 6.드라마, 7.로맨스, 8.범죄, 9.스포츠, 10.액션, 11.역사, 12.코미디, 13.판타지 이 13개의 장르에서 무작위로 번호 1~3개 골라서 그 번호에 해당하는 장르 대답해.0. OOO 형식으로 대답해 다른말은 필요없어"
    response = model_gemini.generate_content(prompt)
    return response.text.strip()

# ✅ PyQt 메인 클래스
class GenrePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Gemini 기반 시놉시스 장르 예측기")
        self.resize(600, 400)

        self.layout = QVBoxLayout()

        self.label = QLabel("시놉시스를 입력하거나 Gemini AI 생성 버튼을 눌러보세요!")
        self.text_edit = QTextEdit()
        self.predict_button = QPushButton("🎯 장르 예측하기")
        self.random_button = QPushButton("🌟 시놉시스 Gemini 생성")
        self.result_label = QLabel("")

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.text_edit)
        self.layout.addWidget(self.predict_button)
        self.layout.addWidget(self.random_button)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

        self.predict_button.clicked.connect(self.predict_genre)
        self.random_button.clicked.connect(self.insert_random_synopsis)

    def predict_genre(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "입력 오류", "시놉시스를 입력해주세요!")
            return
        try:
            x = preprocess_text(text)
            pred = model.predict(x)[0]
            result = {label: float(prob) for label, prob in zip(labels, pred)}
            sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

            top_k = list(sorted_result.items())[:5]
            display = "\n".join([f"{genre}: {score:.2%}" for genre, score in top_k])
            self.result_label.setText(f"<b>🔍 예측 결과:</b><br>{display}")

        except Exception as e:
            QMessageBox.critical(self, "예측 실패", f"에러: {e}")

    def insert_random_synopsis(self):
        try:
            generated = generate_random_synopsis()
            self.text_edit.setPlainText(generated)
        except Exception as e:
            QMessageBox.warning(self, "시놉시스 생성 실패", f"에러: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = GenrePredictor()
    win.show()
    sys.exit(app.exec_())
