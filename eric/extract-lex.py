import os
import json 

done=[]
with open("lex-vers2.json", mode='r', encoding='utf8') as jfile:
    out=json.load(jfile)
for i in out:
    wav=out[i]['ref'].replace('.wrd', '.wav')
    if wav not in done:
        os.system('scp ceos.imag.fr:/home/getalp/leferrae/thesis/corpora/mboshi-french-parallel-corpus/full_corpus_newsplit/all/'+wav+' ./data/')
        done.append(wav)