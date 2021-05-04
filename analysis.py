import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()
data = pd.read_csv('translations.csv', sep='\t', index_col=0)

e = pd.DataFrame()
def process(row):
    for w in row:
        e.append([w])
print('tokenizing')
eng_tok = data.apply(lambda row: process(tokenizer.tokenize(row.ENGLISH)), axis=1)
heb_tok = data.apply(lambda row: process(tokenizer.tokenize(row.ENGLISH)), axis=1)
print('making df')
trans_tok = pd.DataFrame(data=[eng_tok,heb_tok],columns=['ENGLISH','HEBREW'])
print(trans_tok)