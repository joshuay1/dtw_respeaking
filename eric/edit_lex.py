import json
import os
from pydub import AudioSegment
from pydub.playback import play

def crop_audio(deb, fin, path, mot, out, word):
    ext = AudioSegment.from_wav(path)
    ext = ext[deb:fin]
    again=True
    print(path)
    while again==True:
        print(word)

        play(ext)
        print("marche pas")
        again=input("again?(y/n)")
        if again=="n":
            again=False
        else:
            again=True
    ext.export(out + "m"+str(mot), format="wav")

ind = 12

audio = '/home/leferrae/thesis/audios/'
ref = '/home/leferrae/thesis/crops/lex100/spoken_lex100.json'
with open(ref, mode='r', encoding='utf8') as jsfile:
    lex = json.load(jsfile)

elt = lex[ind-1]
bad = elt['ref']
mb = '/home/leferrae/thesis/mboshi-french-parallel-corpus/full_corpus_newsplit/all/'
timefold = '/home/leferrae/thesis/wrd/'
test=''
for f in os.listdir(timefold):
    if f!= bad.replace('.wav', '.wrd'):
        print("processing : {}".format(f))
        with open(timefold + f) as times:
            times_op=times.read()
        if elt['mboshi'] in times_op.lower():
            div_time=times_op.split("\n")
            for line in div_time:
                if elt['mboshi'] in line.lower():
                    parts=line.split()
                    deb=float(parts[0])*1000
                    fin=float(parts[1])*1000
                    audio_file=f.replace(".wrd", ".wav")
                    crop_audio(deb, fin, audio+audio_file, ind, '/home/leferrae/thesis/new_crops/', elt['mboshi'])
                    test=input("is that correct?")
                    if test=="y":
                        elt['ref'] = os.path.basename(audio_file)
                        lex[ind-1] = elt
                        with open(ref, mode='w', encoding='utf8') as jfile:
                            json.dump(lex, jfile, ensure_ascii=False)
                        break
        if test=='y':
            break
print(elt)