import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from data_clean_general import tokenizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

fn = 'clean_combined_aapl.csv'


def clean(data):
  if isinstance(data, str):
    data = pd.read_csv(fn)
  elif isinstance(data, pd.DataFrame):
    pass
  del data['source']
  del data['topic']
  del data['url']
  data['sentiment']=0
  data["headline"] = data["headline"].apply(lambda x: '' if str(x) == 'nan' else x)
  data["bodytext"] = data["bodytext"].apply(lambda x: '' if str(x) == 'nan' else x)
  data["message"] = data["headline"] + data["bodytext"]
  del data['headline']
  del data['bodytext']
  data['message'] = data['message'].apply(lambda x: tokenizer(x) if str(x) != 'nan' else '')
  data['message_length'] = data['message'].apply(len)
  data = data[(data['message_length'] > 3)]
  return data
