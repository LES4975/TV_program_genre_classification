import sys
import re
import numpy as np
import google.generativeai as genai
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QMessageBox
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
from konlpy.tag import Okt



# ✅ Gemini API 키 설정
genai.configure(api_key="  ")    # API 키 넣기
model_gemini = genai.GenerativeModel('gemini-2.0-flash')    # 원하는 gemini 모델 넣기

# ✅ 모델 및 토크나이저, 인코더 불러오기
model = load_model('./models/multilabel_classification_model_0.5342.h5')
with open('./models/token_max_273.pickle', 'rb') as f:
    token = pickle.load(f)
with open('./models/encoder_multilabel.pickle', 'rb') as f:
    mlb = pickle.load(f)

labels = mlb.classes_
okt = Okt()

# ✅ 전처리 함수
def preprocess_text(text):
    text = re.sub('[^가-힣]', ' ', text)  # 한글 외 문자 제거
    words = okt.morphs(text, stem=True)  # 형태소 분석
    words = [w for w in words if len(w) > 1]  # 한 글자 단어 제거
    joined = ' '.join(words)  # 문자열로 다시 조합
    seq = token.texts_to_sequences([joined])  # 정수 시퀀스로 변환
    pad = pad_sequences(seq, maxlen=273)  # 길이 맞춰 패딩
    return pad



# ✅ PyQt UI 클래스
class GenrePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.last_genre_history = []    # 최근 장르 저장 (중복 방지)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("🎬 Gemini 기반 시놉시스 장르 예측기")
        self.resize(650, 500)
        self.layout = QVBoxLayout()

        # 장르 출력 텍스트
        self.generated_label = QLabel("🧠 Gemini가 고른 장르:")
        self.generated_text = QTextEdit()
        self.generated_text.setReadOnly(True)
        self.generated_text.setFixedHeight(40)  # 장르만 출력할 거라 작게 설정

        # 시놉시스 입력 텍스트
        self.synopsis_label = QLabel("📝 예측할 시놉시스:")
        self.text_edit = QTextEdit()

        # 버튼 & 결과 출력
        self.predict_button = QPushButton("🎯 장르 예측하기")
        self.random_button = QPushButton("🌟 Gemini로 시놉시스 생성")
        self.result_label = QLabel("")  # 예측 결과 표시

        # 배치
        self.layout.addWidget(self.generated_label)
        self.layout.addWidget(self.generated_text)
        self.layout.addWidget(self.synopsis_label)
        self.layout.addWidget(self.text_edit)
        self.layout.addWidget(self.predict_button)
        self.layout.addWidget(self.random_button)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

        # 이벤트 연결
        self.predict_button.clicked.connect(self.predict_genre)
        self.random_button.clicked.connect(self.insert_random_synopsis)

    # ✅ Gemini 시놉시스 생성 함수
    def generate_random_synopsis(self):
        # 최근 생성된 장르 모음 (중복 제거)
        exclude_genres = set(g for genre_list in self.last_genre_history for g in genre_list)
        exclude_str = ', '.join(exclude_genres) if exclude_genres else "없음"

        # Gemini 프롬프트 구성
        prompt = (
            f"1.Reality TV, 2.SF, 3.가족, 4.공포, 5.다큐멘터리, 6.드라마, 7.로맨스, 8.범죄, "
            f"9.스포츠, 10.액션, 11.역사, 12.코미디, 13.판타지 "
            f"이 13개 장르 중에서 이전에 사용된 장르 ({exclude_str})는 제외하고, "
            "무작위로 1~4개를 골라 그걸 기반으로 3문장짜리 시놉시스를 만들어줘. "
            "맨 처음 줄에는 선택한 장르를 적고, 그 아래에는 시놉시스를 적어줘. 그 외에는 아무 말도 하지 마."
        )
        response = model_gemini.generate_content(prompt)
        return response.text.strip()

    def insert_random_synopsis(self):
        try:
            generated = self.generate_random_synopsis()
            lines = generated.strip().split('\n')
            genre_line = lines[0] if len(lines) > 0 else ""
            synopsis_only = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""

            # 상단엔 장르만
            self.generated_text.setPlainText(genre_line)

            # 예측용 시놉시스만 하단에 입력
            self.text_edit.setPlainText(synopsis_only)

            # 최근 장르 갱신 (최대 3개 저장)
            current_genres = [g.strip() for g in re.split('[,|·]', genre_line) if g.strip()]
            self.last_genre_history.append(current_genres)
            if len(self.last_genre_history) > 3:
                self.last_genre_history.pop(0)

        except Exception as e:
            QMessageBox.warning(self, "시놉시스 생성 실패", f"에러: {e}")

    def predict_genre(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "입력 오류", "시놉시스를 입력해주세요!")
            return
        try:
            x = preprocess_text(text)
            pred = model.predict(x)[0]

            # 예측 결과 정렬
            result = {label: float(prob) for label, prob in zip(labels, pred)}
            sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

            # 상위 5개 출력
            top_k = list(sorted_result.items())[:5]
            display = "\n".join([f"{genre}: {score:.2%}" for genre, score in top_k])
            self.result_label.setText(f"<b>🔍 예측 결과:</b><br>{display}")

        except Exception as e:
            QMessageBox.critical(self, "예측 실패", f"에러: {e}")

# ✅ 실행
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = GenrePredictor()
    win.show()
    sys.exit(app.exec_())
