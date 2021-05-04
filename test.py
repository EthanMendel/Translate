from numpy import array, argmax, random, take
import pandas as pd

arr = array([['go', 'geh','CC-BY 2.0 (France) Attribution: tatoeba.org #2877272 (CM) & #8597805 (Roujin)'],
      ['hi', 'hallo','CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #380701 (cburgmer)'],
      ['hi', 'grüß gott','CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #659813 (Esperantostern)'],
      ['no one encouraged her', 'niemand ermutigte sie','CC-BY 2.0 (France) Attribution: tatoeba.org #40334 (Swift) & #370461 (Wolf)'],
      ['no one has that right', 'niemand hat dieses recht','CC-BY 2.0 (France) Attribution: tatoeba.org #2891740 (CK) & #6176735 (sprachensprech)'],
      ['no one has that right', 'dieses recht hat niemand','CC-BY 2.0 (France) Attribution: tatoeba.org #2891740 (CK) & #6176738 (sprachensprech)']])


print(arr[:,0])
print('------------------------------')
print(arr[:,1])

trans = pd.read_csv('translations.csv',sep='\t')
print(trans['ENGLISH'].tolist())
print('------------------')
print(trans['HEBREW'])