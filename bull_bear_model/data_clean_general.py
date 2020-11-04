import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def tokenizer(s):
  clean(s)
  translate_table = dict((ord(char), None) for char in string.punctuation)
  s = s.translate(translate_table)
  tokens = word_tokenize(s)
  filtered_tokens = []
  for word in tokens:
    if word.lower() not in stop_words:
      filtered_tokens.append(word.lower())
  return filtered_tokens

def clean(s):
  s = re.sub(r'http\S+', '', s)
  s = re.sub(r'\$(\w+)', '', s)
  return s