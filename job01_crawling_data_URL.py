from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import datetime

# 크롬 드라이버 설정
options = ChromeOptions()
options.add_argument('--headless') # 브라우저를 안 띄울 것이다
options.add_argument('lang=ko_KR') # 한국어를 사용
options.add_argument('--disable-gpu')           # GPU 비활성화 (Windows에서 필수일 수 있음)
options.add_argument('--window-size=1920x1080') # 가상 브라우저 해상도 지정
options.add_argument('--no-sandbox')            # 보안 모드 비활성화 (리눅스에서 권장됨)
options.add_argument('--disable-dev-shm-usage') # 메모리 이슈 방지
service = ChromeService(executable_path=ChromeDriverManager().install()) # 크롬 드라이버 설치
driver = webdriver.Chrome(service=service, options=options)

# 장르 카테고리
genres = ['역사', '로맨스', '드라마', '판타지', '공포', '스릴러',
          '스포츠', '액션', '코미디', '가족', 'SF', 'Reality TV',
          '범죄', '다큐멘터리']

main_url = "https://www.justwatch.com/kr?exclude_genres=ani,eur,msc,war,wsn"
df = pd.DataFrame()
print("Let's crawling data!")

# 메인 페이지 열기
driver.get(main_url)
print("메인 페이지 열기")
time.sleep(3)

last_height = driver.execute_script("return document.body.scrollHeight") # 스크롤 길이

# 스크롤 내리기
for i in range(10):
    print("스크롤 내리기")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(0.5) # 스크롤을 내린 후 대기

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break  # 더 이상 로딩되는 데이터가 없을 경우 종료
    last_height = new_height


# 상세 페이지로 이동하는 URL 수집하기
# 'title-list-grid__item" 클래스를 가진 div 아래아래에 있는 a 태그의 XPath 저장
a_href = driver.find_elements(By.XPATH, '//div[contains(@class, "title-list-grid__item")]//a')
hrefs = [tag.get_attribute('href') for tag in a_href if tag.get_attribute('href')] # a 태그의 href 속성값 저장

# 중복 제거
hrefs = list(set(hrefs))

# 결과 출력 (일부만 출력)
print(f"수집한 영화 URL 개수: {len(hrefs)}")
for url in hrefs[:10]:
    print(url)

# 파일로 저장
with open("justwatch_video_url.txt", "w", encoding="utf-8") as f:
    for url in hrefs:
        f.write(url + "\n")