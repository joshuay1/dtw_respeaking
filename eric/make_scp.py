import os
root = '/home/getalp/leferrae/thesis/corpora/Kunwinku-speech/full_corpus_split/wav/'
with open('wav.scp', mode='w', encoding='utf8') as scp:
    for i in os.listdir(root):
        if '.wav' in i:
            scp.write(i.replace('.wav', '')+' '+root+i+'\n')

with open('utt2spk', mode='w', encoding='utf8') as utt:
    for i in os.listdir(root):
        if '.wav' in i:
            f=i.split('_')
            utt.write(i.replace('.wav', '')+' '+f[0]+'\n')
with open('segments', mode = 'w', encoding='utf8') as seg:
    for i in os.listdir(root):
        if '.wav' in i:
            seg.write(i.replace('.wav', '')+'\n')
# from pydub import AudioSegment
# ext = AudioSegment.from_wav('/home/leferrae/thesis/mboshi-french-parallel-corpus/full_corpus_newsplit/all/martial_2015-09-07-14-53-15_samsung-SM-T530_mdw_elicit_Dico19_39.wav')
# ext = ext[570:890]
# ext.export('./m35.wav', format='wav')