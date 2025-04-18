from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

# 셀레니움 브라우저 설정
options = Options()
options.add_argument('--start-maximized')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# JustWatch 한국 페이지 (일부 장르 제외)
url = 'https://www.justwatch.com/kr?exclude_genres=ani,eur,msc,war,wsn'
driver.get(url)
time.sleep(2)  # 페이지 로딩 대기


# 상위 10개 프로그램 링크 수집
program_elements = driver.find_elements(By.CSS_SELECTOR, 'a.title-list-grid__item--link')[:5]
hrefs = [elem.get_attribute('href') for elem in program_elements]

video_info = []

# 10개 프로그램 상세 정보 크롤링
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
            genre = genres_list[-1] if genres_list else ""
        except NoSuchElementException:
            genres_list = []

        video_info.append({
            "title": title,
            "synopsis": synopsis,
            "genres": genre
        })

        print(f"✅ {i+1}/10: {title} | {genre}")
    except Exception as e:
        print(f"❌ {i+1}/10 에러: {e}")
        continue

driver.quit()

# 결과 저장
df = pd.DataFrame(video_info, columns=["title", "synopsis", "genres"])
df.to_csv('justwatch_test_10.csv', index=False, encoding='utf-8-sig')
print("🎉 테스트 완료! justwatch_test_10.csv 저장됨")
