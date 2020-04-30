# sparse transcription with human in the loop with dtw.
in main.py change the paths.
call launch with python3 main.py --rep (mfcc, vtln, plp or wtv) --norm (cmvn or none) -- feat (kind of corpus you are using)\\

# dependencies:

Shennong: kaldi based features extractors. Compulsory for vltn transformation and plp. 
https://anaconda.org/coml/shennong
Pykaldi : used by Shennong : https://anaconda.org/pykaldi/pykaldi

progess: https://pypi.org/project/progress/

torch: used for wav2vec feature extractor: https://pytorch.org/

praatio: used to extract the information from TextGrid files: https://pypi.org/project/praatio/

pydub: Used to clip out the hits : https://pypi.org/project/pydub/
