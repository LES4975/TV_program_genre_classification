# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service as ChromeService
# from selenium.webdriver.chrome.options import Options as ChromeOptions
# from webdriver_manager.chrome import ChromeDriverManager
# import time
# import pandas as pd
# import datetime
#
# category = [
#     "Action",         # 액션
#     "Comedy",         # 코미디
#     "Crime",          # 범죄
#     "Documentary",    # 다큐멘터리
#     "Drama",          # 드라마
#     "Fantasy",        # 판타지
#     "History",        # 역사
#     "Horror",         # 공포
#     "Mystery",        # 미스터리
#     "Romance",        # 로맨스 / 멜로
#     "Science-Fiction",# SF / 공상 과학
#     "Thriller",       # 스릴러
#     "Reality",        # 리얼리티 예능
#     "Sport",          # 스포츠
#     "Family"          # 가족
# ]
# options = ChromeOptions()
# options.add_argument('lang=ko_KR')
#
# service = ChromeService(executable_path=ChromeDriverManager().install())
# driver = webdriver.Chrome(service=service, options=options)
#
# url ='https://www.justwatch.com/kr?exclude_genres=ani,eur,msc,war,wsn'
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

# 📌 크롬 옵션 설정
options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 📌 대상 사이트
url = 'https://www.justwatch.com/kr?exclude_genres=ani,eur,msc,war,wsn'
driver.get(url)
time.sleep(3)

# 📌 스크롤 다운 (끝까지)
prev_height = 0
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    curr_height = driver.execute_script("return document.body.scrollHeight")
    if curr_height == prev_height:
        break
    prev_height = curr_height

# 📌 프로그램 링크 수집
program_elements = driver.find_elements(By.CSS_SELECTOR, 'a.title-list-grid__item--link')
hrefs = [elem.get_attribute('href') for elem in program_elements]
print(f"총 {len(hrefs)}개 수집됨")

# 📌 크롤링 시작
video_info = []

for i, url in enumerate(hrefs):
    try:
        driver.get(url)
        time.sleep(2)

        # 제목
        try:
            title = driver.find_element(By.XPATH, '//h1[contains(@class, "title-detail-hero__details__title")]').text.strip()
        except NoSuchElementException:
            title = ""

        # 시놉시스
        try:
            synopsis = driver.find_element(By.XPATH, '//p[contains(@class, "text-wrap-pre-line mt-0")]').text.strip()
        except NoSuchElementException:
            synopsis = ""

        # 장르
        try:
            genre_tags = driver.find_elements(By.XPATH, '//div[contains(@class, "poster-detail-infos__value")]//span')
            genres_list = [g.text.strip() for g in genre_tags if g.text.strip()]
        except NoSuchElementException:
            genres_list = []

        # 결과 저장
        video_info.append({
            "title": title,
            "synopsis": synopsis,
            "genres": genres_list,
            "genre_count": len(genres_list),
            "url": url
        })

        # 중간 저장
        if i % 100 == 0 and i != 0:
            df_temp = pd.DataFrame(video_info)
            df_temp.to_csv(f'justwatch_temp_{i}.csv', index=False, encoding='utf-8-sig')
            print(f"💾 {i}개 저장됨")

        print(f"✅ {i+1}/{len(hrefs)} 완료: {title} ({', '.join(genres_list)})")

    except Exception as e:
        print(f"❌ {i+1}번째 실패: {e}")
        continue

# 📌 최종 저장
df = pd.DataFrame(video_info)
df.to_csv("justwatch_tv_all.csv", index=False, encoding='utf-8-sig')
print("🎉 전체 크롤링 완료! justwatch_tv_all.csv 저장됨")

driver.quit()
