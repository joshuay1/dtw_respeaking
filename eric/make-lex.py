import os
import json
from pydub.playback import play
from pydub import AudioSegment

'''
m = 0
k = 0
a = 0
root="/home/getalp/leferrae/thesis/corpora/"
audio=root+"mboshi-french-parallel-corpus/full_corpus_newsplit/all/"
align = root+"wrd/"
out = {}
for wrd in os.listdir(align):
    with open(align+wrd, mode="r", encoding="utf8") as fwrd:
        rwrd=fwrd.read().split("\n")
    for lline in rwrd:
        line=lline.split()
        if len(line)>1:
            w=line[2].lower()
            if w not in out and len(w)>2 and line[2]!='SIL':
                if 'martial' in wrd and m<33:
                    out[w] = {'ref': wrd, 'beg' : line[0], 'end' : line[1]}
                    m+=1
                elif 'kouarata' in wrd and k<33:
                    out[w] = {'ref': wrd, 'beg' : line[0], 'end' : line[1]}
                    k+=1
                elif 'abiayi' in wrd and a<34:
                    out[w] = {'ref': wrd, 'beg' : line[0], 'end' : line[1]}
                    a+=1
with open("lex-vers2.json", mode='w', encoding='utf8') as jfile:
    json.dump(out, jfile)
'''
cpt=1
out=[]
with open('lex-vers2.json', mode='r', encoding='utf8') as jfile:
    ref=json.load(jfile)
for elt in ref:
    wav_fn=ref[elt]['ref'].replace('wrd', 'wav')
    deb=float(ref[elt]['beg'])*1000
    fin=float(ref[elt]['end'])*1000
    print(wav_fn)
    ext = AudioSegment.from_wav('./data/'+wav_fn)
    seg = ext[deb:fin]
    move=''
    while move!='v':
        play(seg)
        print('word : {}'.format(elt))
        move=input('shift before(b)? after(a) listen again (l) or valid (v)')
        if move=='b':
            shift=input('how much in second')
            seg= ext[deb-int(shift):fin]
        elif move=='a':
            shift=input('how much in second')
            seg= ext[deb:fin+int(shift)]
        elif move=='v':
            seg.export('./crops/m{}.wav'.format(cpt), format='wav')
            out.append({'ref':wav_fn, 'mboshi':elt, "french":"unknown", "crop":"/home/getalp/leferrae/thesis/corpora/crops/lex-vers2/m{}.wav".format(cpt)})
            cpt+=1

        elif move=='l':
            play(seg)
        else:
            print('enter a valid answer')
with open('lex100-vers2.json', mode='w', encoding='utf8') as jfile:
    json.dump(out, jfile, ensure_ascii=False)