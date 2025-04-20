import pandas as pd
import glob

from tensorflow.python.ops.metrics_impl import false_negatives

data_dir = './crawling_data/'
data_path = glob.glob(data_dir + '*.csv')
print(data_path)
df = pd.DataFrame()
for path in data_path:
    df_section = pd.read_csv(path)
    df = pd.concat([df, df_section], ignore_index=True)
df.info()
print(df.head())
df.to_csv('./crawling_data/data.csv', index = False)