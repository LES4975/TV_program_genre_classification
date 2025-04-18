from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime


category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']
df_titles = pd.DataFrame()

for i in range(len(category)):
    url = 'https://news.naver.com/section/10{}'.format(i)

    response = requests.get(url)
    # print(list(response))

    soup = BeautifulSoup(response.text, features='html.parser')
    # print(list(soup))

    title_tags = soup.select('.sa_text_strong')
    # print(title_tags)

    title = title_tags[0].text
    # print(title)

    titles = []
    for tag in title_tags:
        titles.append(tag.text)

    # print(titles)

    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_section_titles['category'] = category[i]

    df_titles = pd.concat([df_titles, df_section_titles],
                          axis=0, ignore_index=True)

print(df_titles.tail())

print(df_titles['category'].value_counts())
df_titles.to_csv('./crawling_data/naver_headline_news_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)