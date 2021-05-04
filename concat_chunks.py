import pandas as pd
import csv

eng = pd.read_csv("eng_sentences.tsv", sep="\t", names=["SENTENCE_ID","LANGUAGE","TEXT"])
heb = pd.read_csv("heb_sentences.tsv", sep="\t", names=["SENTENCE_ID","LANGUAGE","TEXT"])

# full_links = pd.DataFrame(columns=['SENTENCE_ID','TRANSLATION_ID'])
# for i in range(183):
#     print(f'file number {i} total links {len(full_links.index)}')
#     chunk = pd.read_csv(f'needed_links_{i}.csv',sep='\t')
#     for j,l in chunk.iterrows():
#         #print(l)
#         full_links=full_links.append(pd.DataFrame({'SENTENCE_ID':[l['SENTENCEE_ID']],'TRANSLATION_ID':[l['TRANSLATION_ID']]}),ignore_index=True)
# full_links.to_csv(f'needed_links_full.csv',sep='\t')
full_links = pd.read_csv('needed_links_full.csv',sep='\t')

ids = []
translations = pd.DataFrame(columns=['ENGLISH','HEBREW'])
for i,l in full_links.iterrows():
    if i%1000==0:
        print(f'index {i}')
    l1 = eng.loc[eng['SENTENCE_ID'] == l['SENTENCE_ID']]
    l2 = heb.loc[heb['SENTENCE_ID'] == l['TRANSLATION_ID']]
    if not l1.empty and not l2.empty:
        id1 = l1.iloc[0]['SENTENCE_ID']
        id2 = l2.iloc[0]['SENTENCE_ID']
        if not id1 in ids and not id2 in ids:
            ids.append(id1)
            ids.append(id2)
            t1 = ''.join(p for p in l1.iloc[0]['TEXT'] if p not in ['.','?',',','!',':',';'])
            t2 = ''.join(p for p in l2.iloc[0]['TEXT'] if p not in ['.','?',',','!',':',';'])
            translations = translations.append(pd.DataFrame({'ENGLISH':[t1],'HEBREW':[t2]}),ignore_index=True)
translations.to_csv('tatoeba_translations.csv',sep='\t')


