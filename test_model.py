import string
import re
from numpy import array, argmax, random, take
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, RepeatVector, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

pd.set_option('display.max_colwidth', 200)

eng_heb = pd.read_csv('translations.csv',sep='\t')
englishList = eng_heb['ENGLISH'].tolist()
hebrewList = eng_heb['HEBREW'].tolist()

englishList = englishList[:50000]
hebrewList = hebrewList[:50000]
# empty lists
eng_l = []
eng_length = 0
heb_l = []
heb_length = 0

# populate the lists with sentence lengths
for i in englishList:
    eng_l.append(len(i.split(" ")))
    l=len(i.split(" "))
    if(l>eng_length):
        eng_length=l
print('English Longest Sentense: %d' % eng_length)


for i in hebrewList:
    heb_l.append(len(i.split(" ")))
    l=len(i.split(" "))
    if(l>heb_length):
        heb_length=l
print('Hebrew Longest Sentense: %d' % heb_length)

# function to build a tokenizer
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# prepare english tokenizer
eng_tokenizer = tokenization(englishList)
eng_vocab_size = len(eng_tokenizer.word_index) + 1

print('English Vocabulary Size: %d' % eng_vocab_size)

# prepare Hebrew tokenizer
heb_tokenizer = tokenization(hebrewList)
heb_vocab_size = len(heb_tokenizer.word_index) + 1

print('Hebrew Vocabulary Size: %d' % heb_vocab_size)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

from sklearn.model_selection import train_test_split
eng_train, eng_test = train_test_split(englishList, test_size=0.2, random_state = 12)
heb_train, heb_test = train_test_split(englishList, test_size=0.2, random_state = 12)

# prepare training data
# heb as input and eng as output
trainX = encode_sequences(heb_tokenizer, heb_length, heb_train)
trainY = encode_sequences(eng_tokenizer, eng_length, eng_train)

# prepare validation data
testX = encode_sequences(heb_tokenizer, heb_length, heb_test)
testY = encode_sequences(eng_tokenizer, eng_length, eng_test)

# Load NMT model
model1 = load_model('model.h1.12_nov_20')# trained for 6 epochs locally using the first 50,000 sentence pairs (train + test)
preds1 = model1.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))
# model2 = load_model('model.h1.20_nov_20')# trained for 30 epochs on mamba using the first 50,000 sentence pairs (train + test)
# preds2 = model2.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))
# model3 = load_model('model.h1.24_nov_20')# trained for 90 epochs on mamba using the first 50,000 sentence pairs (train + test)
# preds3 = model3.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))
# model4 = load_model('model.h1.03_dec_20')# trained for __ epochs on mamba using the whole 18 million pairs (train + test)
# preds4 = model4.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))

def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None

def get_preds(preds):
    preds_text = []
    for i in preds:
        temp = []
        for j in range(len(i)):
            t = get_word(i[j], eng_tokenizer)
            if j > 0:
                if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
                    temp.append('')
                else:
                    temp.append(t)
                
            else:
                if(t == None):
                    temp.append('')
                else:
                    temp.append(t)            
            
        preds_text.append(' '.join(temp))
    return preds_text

# convert predictions into text (English)
preds_text = get_preds(preds1)
# preds_text2 = get_preds(preds2)
# preds_text3 = get_preds(preds3)
# preds_text4 = get_preds(preds3)

""" figurer out how to concatinate all of them """
# get bleu and meteor scores
nltk.download('wordnet')
bleu_score = []
met_score = []
for i in range(len(eng_test)):
    eng = eng_test[i].split(" ")
    pred = preds_text[i].split(" ")
    bleu_score.append(sentence_bleu(eng,pred))
    met_score.append(meteor_score(eng,preds_text[i]))
length = len(bleu_score)
print(f'Average BLEU score: {sum(bleu_score)/length}')
# Average BLEU score: 1.8345392759940913e-159
print(f'Average METEOR score: {sum(met_score)/length}')
#Average METEOR score: 0.26234439726940284

# build dataframe and look at model results
pred_df = pd.DataFrame({'actual' : eng_test, 'predicted' : preds_text, 'bleu' : bleu_score, 'meteor' : meteor_score})
pd.set_option('display.max_colwidth', 200)
pred_df.sample(15)