import pandas as pd
import csv
import threading

def process_chunk(chunk, chunk_num):
    added_count = 0
    links = pd.DataFrame(columns = ['SENTENCE_ID','TRANSLATION_ID'])
    for i,l in chunk.iterrows():
        print(f'chunk {chunk_num} index {i} added {added_count}')
        l1 = eng.loc[eng['SENTENCE_ID'] == l['SENTENCE_ID']]
        l2 = heb.loc[heb['SENTENCE_ID'] == l['TRANSLATION_ID']]
        if not l1.empty and not l2.empty:
            id1 = l1.iloc[0]['SENTENCE_ID']
            id2 = l2.iloc[0]['SENTENCE_ID']
            links=links.append(pd.DataFrame({'SENTENCE_ID':[id1],'TRANSLATION_ID':[id2]}),ignore_index=True)
            #print(len(links.index))
            added_count+=1
    links.to_csv(f'needed_links_{chunk_num}.csv',sep='\t')
    

eng = pd.read_csv("eng_sentences.tsv", sep="\t", names=["SENTENCE_ID","LANGUAGE","TEXT"])
heb = pd.read_csv("heb_sentences.tsv", sep="\t", names=["SENTENCE_ID","LANGUAGE","TEXT"])
link_chunks = pd.read_csv("links.csv", sep="\t", names=["SENTENCE_ID","TRANSLATION_ID"], chunksize=100000)
for i, chunk in enumerate(link_chunks):
    threading.Thread(target=process_chunk,args=(chunk,i)).start()
