# Reindex item feature ID with IDs generated in rating2inter.py
import os
import pandas as pd
import gzip
os.chdir('/home/minhle/CaMRec/data')
os.getcwd()

# ultis functions
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# load item mapping
i_id_mapping = 'i_id_mapping.csv'
df = pd.read_csv(i_id_mapping, sep='\t')
meta_file = '../raw_data/meta_Baby.json.gz'

meta_df = getDF(meta_file)

# remapping
map_dict = dict(zip(df['asin'], df['itemID']))

meta_df['itemID'] = meta_df['asin'].map(map_dict)
meta_df.dropna(subset=['itemID'], inplace=True)
meta_df['itemID'] = meta_df['itemID'].astype('int64')
#meta_df['description'] = meta_df['description'].fillna(" ")
meta_df.sort_values(by=['itemID'], inplace=True)

ori_cols = meta_df.columns.tolist()

ret_cols = [ori_cols[-1]] + ori_cols[:-1]

ret_df = meta_df[ret_cols]
# dump
ret_df.to_csv(os.path.join('./', 'meta-baby.csv'), index=False)