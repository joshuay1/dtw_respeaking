import os
import json

with open('/home/leferrae/spoken_lex100.json', mode='r', encoding='utf8') as jfile:
    lex = json.load(jfile)
real = []
for i in lex:
    real.append(i['mboshi'])
recap={}
root='/home/leferrae/thesis/mboshi-french-parallel-corpus/full_corpus_newsplit/all/'
for f in os.listdir(root):
    if '.mb.cleaned' in f:
        with open(root+f, mode='r', encoding='utf8') as fic:
            s = fic.read()
            for mot in real:
                if mot in s:
                    if mot in recap:
                        recap[mot]+=1
                    else:
                        recap[mot]=1
tot=0
for i in real[0:20]:
    tot+=(recap[i])
print(tot)
sortDict=sorted(recap.items(), key=lambda x: x[1], reverse=True)
