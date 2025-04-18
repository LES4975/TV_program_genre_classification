import pandas as pd
import glob

data_dir = './crawling_data/'
data_path = glob.glob(data_dir + '*.*')
print(data_path)
df = pd.DataFrame()
for path in data_path:
    df_section = pd.read_csv(path)
    df = pd.concat([df, df_section],
                   ignore_index=True)
df.info()
print(df.head())
df.to_csv('./crawling_data/news_titles.csv', index=False)