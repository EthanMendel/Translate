import pandas as pd
import csv
from nltk.tokenize import TweetTokenizer
import nltk

# wiki_en = open('Wikipedia.en-he.en.txt', 'r')
# wiki_en = wiki_en.readlines()
# wiki_he = open('Wikipedia.en-he.he.txt', 'r')
# wiki_he = wiki_he.readlines()
#
# for i, (en, he) in enumerate(zip(wiki_en, wiki_he)):
#     if not en.isascii():
#         wiki_en.pop(i)
#         wiki_he.pop(i)
#         continue
#     wiki_en[i] = ''.join(c for c in en if c not in ['.', '?', ',', '!', ':', ';'])
#     wiki_en[i] = wiki_en[i].replace('\n', '')
#     wiki_he[i] = ''.join(c for c in he if c not in ['.', '?', ',', '!', ':', ';'])
#     wiki_he[i] = wiki_he[i].replace('\n', '')
#
# wiki = pd.concat([pd.DataFrame(wiki_en, columns=['ENGLISH']), pd.DataFrame(wiki_he, columns=['HEBREW'])], axis=1)
# wiki.to_csv('wiki.tsv', sep='\t')
#
#
# wiki = pd.read_csv('wiki.tsv', sep='\t', index_col=0)
data = pd.read_csv('tatoeba_translations.csv', sep='\t', index_col=0)
# data = pd.concat([wiki, translations], ignore_index=True)
# print(data)
# data.to_csv('data.tsv', sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\')



eng_vocab = pd.DataFrame(columns=['ENGLISH'])
heb_vocab = pd.DataFrame(columns=['HEBREW'])

tokenizer = TweetTokenizer()

e = pd.DataFrame()
def process(row):
    for w in row:
        e.append([w])

eng_vocab = data.apply(lambda row: process(tokenizer.tokenize(row.ENGLISH)), axis=1)
# heb_vocab = data.apply(lambda row: process(tokenizer.tokenize(row.HEBREW)), axis=1)

# eng_freq = nltk.FreqDist(eng_vocab)
# heb_freq = nltk.FreqDist(heb_vocab)

# for i, s in data.iterrows():
#     print(s)
#     eng_words = nltk.word_tokenize(s.ENGLISH)
#     eng_vocab.append(eng_words)
#
#     heb_words = nltk.word_tokenize(s.HEBREW)
#     heb_vocab.append(heb_words)

eng_vocab.to_csv('eng_vocab.tsv', sep='\t')
heb_vocab.to_csv('heb_vocab.tsv', sep='\t')

print(eng_vocab)
print(heb_vocab)
