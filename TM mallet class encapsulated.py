# Topic Modeling Version 1.3
# Class name is LdaModelBuilding, in which data readin, tokenization, lem&stem,
# stopwords, and model building are all integrated.

# Input arguments are path (where the raw data located), query date, and topic numbers
# among these arguments, the topic number = 20 is optimized thru hyper-parameter tuning
# the query date shall be given when defining an instance (from the web calendar selection)
# function main() is an example to show how to invoke an instance.

# Import libraries
print('Importing libraries...')

import pandas as pd
import gensim
from gensim import corpora as corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet

import nltk
nltk.download('wordnet')
from matplotlib import pyplot as plt
from pprint import pprint

import re
import datetime
from datetime import datetime as dtime

import json

def toDate(item):
    match = re.search(r'\d{4}-\d{1,2}-\d{1,2}', item)
    date = dtime.strptime(match.group(), '%Y-%m-%d').date()
    return date

# Encapsulate all processes into a class named LdaModelBuilding
class LdaModelBuilding():
    # Initialization function
    def __init__(self, path, Q_date, topic_number):
        self.path = path
        self.Q_date = Q_date
        self.topic_number = topic_number
    
    # Data reading function
    def dataread(self):
        print('Reading CSV data ...')
        dataframe = pd.read_csv(self.path)
        dataframe['dateTime'] = dataframe['date'].map(toDate)
        date01 = dtime.strptime(self.Q_date, '%Y-%m-%d').date()
        print(date01)
        date02 = date01 - datetime.timedelta(days = 90)
        print(date02)
        dt = dataframe[(dataframe['dateTime']>=date02)&(dataframe['dateTime']<=date01)]
        print(dt.shape)
        return dt

    # Tokenization function and punctuations removal
    def token(self,sents):
        print('Tokenizing sentences...')
        for sent in sents:
            yield(gensim.utils.simple_preprocess(str(sent),deacc = True))

    # Lemmatize function
    def lem_stem(self,item):
        lemma = WordNetLemmatizer()
        return lemma.lemmatize(item)

    # Eliminate stopwords or words' length less than or eqaul to 3
    def stpwds(self,item):
        res = []
        for wd in item:
            if len(wd) > 3:
                res.append(self.lem_stem(wd))
        return res

    def ldaModeling(self):
        dt = self.dataread()
        data = dt.bodytext.values.tolist()
        data_token = list(self.token(data))
        data_proc = []
        print('Removing stopwords...')
        for text in data_token:
            data_proc.append(self.stpwds(text))
        print(data_proc[:5])
        print('Creating dictionary and corpus for LDA Modeling...')
        id2word = corpora.Dictionary(data_proc)
        corpus = [id2word.doc2bow(text) for text in data_proc]
        print(corpus[:1])
        mallet_path = r'C:\mallet\mallet-2.0.8\bin\mallet.bat'
        print('Building LDA Mallet model...')
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path,
                                                     corpus=corpus,
                                                     num_topics=self.topic_number,
                                                     id2word=id2word)
        tp = ldamallet.show_topics(formatted=False)
        pprint(tp)
        
        results = []

        for item in tp:
            rank = item[0];
            topicsArray = '';
            for kit in item[1]:
                topicsArray = topicsArray + ' ' + kit[0]
            results.append((str(rank),topicsArray))

        js = dict(results)
        print(js)
        file_name = r'D:\vdata\SA\phaseII\topic.json'
        try:
            with open(file_name,'w',encoding='utf-8') as fn:
                json.dump(js,fn)
        except IOError as e:
            print(e)

        coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_proc, dictionary=id2word, coherence='c_v')
        coherence_ldamallet = coherence_model_ldamallet.get_coherence()
        print('\nCoherence Score: ', coherence_ldamallet)
        print('\nNumber of Topics:', self.topic_number)
        

        
# Define main function
def main():
    print('Executing main()..')
    # Define an instance
    lda = LdaModelBuilding(r'D:\vdata\SA\phaseII\sentiment_data\clean_combined_aapl.csv',
                           '2020-05-15',
                           20)
    lda.ldaModeling()

# Main loop
if __name__ == '__main__':
    main()
