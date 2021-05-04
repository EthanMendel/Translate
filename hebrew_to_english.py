"""We are evaluating our model using the BLEU score and METEOR score of the translations because they provide good
metrics for evaluating the effectiveness of translation models. BLEU score evaluates the quality of the text translated
and METEOR score is a mean of recall and percision scores with recall more hevily weighted."""
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
pd.set_option('display.max_colwidth', 200)

eng_heb = pd.read_csv('translations.csv',sep='\t')
englishList = eng_heb['ENGLISH'].tolist()
hebrewList = eng_heb['HEBREW'].tolist()

# englishList = englishList[:50000]
# hebrewList = hebrewList[:50000]

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

# build NMT model
def build_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model

model = build_model(heb_vocab_size, eng_vocab_size, heb_length, eng_length, 512)
rms = optimizers.RMSprop(lr=0.001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

filename = 'model.h1.27_nov_20'
checkpoint = ModelCheckpoint(filepath=filename, monitor='val_loss', verbose=2, save_best_only=True, mode='min')

history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), 
          epochs=90, batch_size=128, 
          validation_split = 0.2,
          callbacks=[checkpoint], verbose=1)