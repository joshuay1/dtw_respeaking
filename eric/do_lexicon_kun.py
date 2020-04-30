import os
from pydub import AudioSegment
import json
import io
from pydub.playback import play
from praatio import tgio
#This script exctract in a semi-automatic way a spoken lexicon from the corpus. It first extract the most frequent words
#avoinding the monosyllabics. It then propose the 5 most propable translation (asking confirmation). It crop an audio
#containing the word and ask the user if it is a good example of the word (playing it)
def get_freq(root):
    '''function which take the Mboshi transcription files and return a list of tuples associating a word
    to its frequency'''

    fulllist={}
    files = os.listdir(root)
    for file in files:
        if ".txt" in file:
            with open(root+file, mode='r', encoding='utf8') as text:
                read=text.read()

                textlist=read.split()
            for r in textlist:
                w=r.replace(',','').lower()
                w=w.replace('?', '')
                w=w.replace('"', '')
                w=w.replace('.', '')

                if len(w)>3:
                    if w not in fulllist:
                        fulllist[w]=1
                    else:
                        fulllist[w]+=1
    sortDict=sorted(fulllist.items(), key=lambda x: x[1], reverse=True)
    print(sortDict[0:60])
    return(sortDict[0:60])

def make_crop(word, mot,size):
    root="/home/leferrae/Desktop/"
    timefold=root+"Kunwinku-speech/forced_align/"
    audio=root+"Kunwinku-speech/full_corpus_split/wav/"
    outfold=root+"/Kunwinku-speech/crops/lex{}/".format(size)
    if os.path.isdir(outfold):
        pass
    else:
        os.mkdir(outfold)
    files=os.listdir(timefold)
    #pour chaque fichier de timesteps, j'en cherche un qui contient le mot
    for f in files:
        print("processing : {}".format(f))
        tg = tgio.openTextgrid(timefold+f)
        tier=tg.tierNameList[0]
        for elt in (tg.tierDict[tier].entryList):
            print(word, elt[2])
            if word == elt[2]:
                deb=float(elt[0])*1000
                fin=float(elt[1])*1000
                audio_file=f.replace(".TextGrid", ".wav")
                crop_audio(deb, fin, audio+audio_file, mot, outfold, word)
                test=input("is that correct?")
                if test=="y":
                    return audio_file
                else:
                    print("process next file")


def crop_audio(deb, fin, path, mot, out, word):
    ext = AudioSegment.from_wav(path)
    ext = ext[deb:fin]
    again=True
    while again==True:
        print(word)
        try:
            play(ext)
        except:
            print("marche pas")
        again=input("again?(y/n)")
        if again=="n":
            again=False
        else:
            again=True
    ext.export(out + "m"+str(mot), format="wav")

# crop_audio(630, 1140, '/home/leferrae/Desktop/Kunwinku-speech/full_corpus_split/wav/interview_64.wav', 6,
# '/home/leferrae/Desktop/Kunwinku-speech/crops/lex100/', 'minj')
txt = '/home/leferrae/Desktop/Kunwinku-speech/full_corpus_split/txt/'
end = (get_freq(txt))

id = 1
json_base = []
size_voc = 100
crop_fold = '/home/leferrae/Desktop/Kunwinku-speech/crops/'


for j,i in enumerate(end):
    print(i[0],j)
    audio_ref = make_crop(i[0], id, size_voc)
    json_ref={}
    json_ref["ref"] = audio_ref
    json_ref["kun"] = i
    json_ref["crop"]=crop_fold+"lex{}/m{}".format(size_voc,id)
    json_base.append(json_ref)
    id+=1
print(json_base)
with io.open("../../mboshi/crops/lex{}/spoken_lex{}.json".format(size_voc,size_voc), mode='w', encoding='utf-8') as dump_json_file:
    json.dump(json_base, dump_json_file, ensure_ascii=False)

