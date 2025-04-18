# raw_data = news_headlines.csv
# columns = [title, category]
# jinwo 0,1 / heachan 2,3 / gyeongmin 4,5

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import datetime

category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']
div_category = [4, 5, 4, 4, 4, 4]
options = ChromeOptions()
options.add_argument('lang=ko_KR')

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# change int value by one's part
my_category = [2, 3]

for num in my_category:
    df_titles = pd.DataFrame()
    button_xpath = '//*[@id="newsct"]/div[{}]/div/div[2]'.format(div_category[num])

    url = 'https://news.naver.com/section/10{}'.format(num)
    driver.get(url)
    cnt = 0
    for cnt in range(50):
        time.sleep(1)
        try :
            driver.find_element(By.XPATH, button_xpath).click()
        except :
            print("error of not more contents")

    time.sleep(5)
    titles = []
    for i in range(1, cnt * 6):
        for j in range(1, 7):
            title_path = '//*[@id="newsct"]/div[{}]/div/div[1]/div[{}]/ul/li[{}]/div/div/div[2]/a/strong'.format(div_category[num], i, j)
            try:
                title = driver.find_element(By.XPATH, title_path).text
                # print(title)
                titles.append(title)
            except:
                print('error', i, j)
    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_titles_category = pd.DataFrame(titles, columns=['titles'])

    df_titles_category['category'] = category[num]
    df_titles = pd.concat([df_titles, df_titles_category], axis=0, ignore_index=True)

    df_titles.to_csv('./crawling_data/news_headlines_{}_{}.csv'.format(
        category[num] , datetime.datetime.now().strftime('%Y%m%d')), index=False)
