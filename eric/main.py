# -*- coding: utf-8 -*-

from dtw import DTW
from dtw import grab_data
from dtw import grab_corp
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--norm", type=str, default="smp")
parser.add_argument("--rep", type=str, default="mfcc")
parser.add_argument("--feat", type=str, default="")



args = parser.parse_args()
iteration = 3
quota = 10
max_inst = 5
wind = 0
rep = args.rep # plp, mfcc, wtv, vtln
norm = args.norm #smp, cmvn
feat = args.feat #reduc
#serveur
lang = "english"
root="/home/josh/transcription_dtw"
lexicon = "/home/josh/transcription_dtw/english/top100words.json"
# lexicon = "/corpora/crops/spoken_lex60.json"
# corpus = "/corpora/mboshi-french-parallel-corpus/full_corpus_newsplit/all/"
corpus = "/home/josh/transcription_dtw/english"
# model = '/home/getalp/leferrae/thesis/mb_model/checkpoint_best.pt'
aligned_fold = '/home/josh/transcription_dtw/english'
# aligned_fold'/home/getalp/leferrae/thesis/corpora/Kunwinku-speech/forced_align/'
# #local
# root = "/home/leferrae/Desktop/These"
# lexicon = "/mboshi/crops/lex100/spoken_lex100.json"
# corpus = "/mboshi/mboshi-french-parallel-corpus/full_corpus_newsplit/all"


queries = grab_data(root, lexicon, rep=rep, norm=norm, feat=feat, lang=lang)
data = grab_corp(root, corpus, rep=rep, norm=norm, feat=feat, lang=lang)

dtw_costs=DTW(queries=queries, search=data, norm=norm, rep=rep, limit=True, feat=feat, lang=lang, wind=wind)

for i in range(0,iteration):
    print("iteration {}".format(i+1))
    dtw_costs.recap_queries(cur=i, size_queries=20*(i+1))
    dtw_costs.eval(size_queries=20*(i+1), corp=root+corpus, root=root, cur=i)
    dtw_costs.do_dtw(size_queries=20*(i+1), ite=iteration, cur=i)
    dtw_costs.do_precision(quota=quota, max_inst=max_inst, aligned_fold=aligned_fold)
    dtw_costs.eval(size_queries=20*(i+1), corp=root+corpus, root=root, cur = i)
