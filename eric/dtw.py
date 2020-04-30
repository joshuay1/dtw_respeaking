# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from os import path
from python_speech_features import delta
from python_speech_features import mfcc
from speech_dtw import qbe
from pydub import AudioSegment
import _pickle as pickle
import multiprocessing
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import sys
import os
import json
import io
import tgt
from progress.bar import Bar
from datetime import datetime
from shennong.audio import Audio
from torch.autograd import Variable
sys.path.append("..")
sys.path.append(path.join("..", "utils"))
from shennong.features.processor.mfcc import MfccProcessor
from shennong.features.postprocessor.delta import DeltaPostProcessor
from shennong.features.processor.plp import PlpProcessor
from shennong.features.postprocessor.cmvn import CmvnPostProcessor
import torch
from torch import nn
from fairseq.models.wav2vec import Wav2VecModel
import soundfile as sf
from praatio import tgio
class PretrainedWav2VecModel(nn.Module):

    def __init__(self, fname):
        super().__init__()

        checkpoint = torch.load(fname)
        self.args = checkpoint["args"]
        model = Wav2VecModel.build_model(self.args, None)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        self.model = model

    def forward(self, x):
        with torch.no_grad():
            z = self.model.feature_extractor(x)
            if isinstance(z, tuple):
                z = z[0]
            c = self.model.feature_aggregator(z)
        return z, c

class Prediction():
    """ Lightweight wrapper around a fairspeech embedding model """

    def __init__(self, fname, gpu=0):
        self.gpu = gpu
        self.model = PretrainedWav2VecModel(fname).cuda(gpu)

    def __call__(self, x):
        x = torch.from_numpy(x).float().cuda(self.gpu)
        with torch.no_grad():
            z, c = self.model(x.unsqueeze(0))

        return z.squeeze(0).cpu().numpy(), c.squeeze(0).cpu().numpy()

def orthogonal(shape, scale=1.1):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape).astype('float32')
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return torch.from_numpy(scale * q[:shape[0], :shape[1]])

def make_linear(size_in, size_out):
    """Returns linear layer with orthogonal initialization."""
    M = nn.Linear(size_in, size_out)
    M.weight.data = orthogonal((size_out, size_in))
    M.bias.data = torch.zeros(size_out)
    return M

def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """

    wav, sr = sf.read(fname)
    assert sr == 16e3

    return wav, 16e3

def to_tuple(obj):
    sortie=[]
    for elt in obj:
        elt=tuple(elt)
        sortie.append(elt)
    return tuple(sortie)

def get_mfcc_vtln(wav_fn, f, norm, lang):
    """Return the MFCCs with deltas and delta-deltas for a audio file."""
    ref = os.path.basename(f).replace(".wav", "")
    if not os.path.isfile("warps_{}.pkl".format(lang)):
        if os.path.isfile('warps_{}.txt'.format(lang)):
            factors={}
            with open('warps_{}.txt'.format(lang), mode='r', encoding='utf-8') as opfile:
                wop=opfile.read().split('\n')
                for line in wop:
                    if len(line)>1:
                        l_sp=line.split()
                        factors[l_sp[0]] = float(l_sp[1])
                        print(factors)
            with open('warps_{}.pkl'.format(lang), mode='wb') as opfile:
                pickle.dump(factors, opfile)
        else:
            print('no warp factors found')
            exit()
    with open("warps_{}.pkl".format(lang), mode="rb") as op:
        factors = pickle.load(op)
    warp = float(factors[ref])
    audio = Audio.load(wav_fn)
    processor = MfccProcessor(sample_rate=audio.sample_rate, window_type="hamming",frame_length=0.025, frame_shift=0.01,
                              cepstral_lifter=26.0,low_freq=0, vtln_low=60, vtln_high=7200, high_freq=audio.sample_rate/2)
    d_processor = DeltaPostProcessor(order=2)
    mfcc_static = processor.process(audio, vtln_warp=warp)
    mfcc_deltas = d_processor.process(mfcc_static)
    features = np.float64(mfcc_deltas._to_dict()["data"])
    if norm == "cmvn":
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    return features
def get_plp(wav_fn, norm):
    """Return the MFCCs with deltas and delta-deltas for a audio file."""
    audio = Audio.load(wav_fn)
    processor = PlpProcessor(sample_rate=audio.sample_rate, window_type="hamming",frame_length=0.025, frame_shift=0.01,
                              low_freq=0, vtln_low=60, vtln_high=7200, high_freq=audio.sample_rate/2)
    plp_static = processor.process(audio, vtln_warp=1.0)
    d_processor = DeltaPostProcessor(order=2)
    plp_deltas = d_processor.process(plp_static)
    features = np.float64(plp_deltas._to_dict()["data"])
    if norm == "cmvn":
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    return features

def get_mfcc_dd(wav_fn, norm):
    """Return the MFCCs with deltas and delta-deltas for a audio file."""
    audio = Audio.load(wav_fn)
    processor = MfccProcessor(sample_rate=audio.sample_rate, window_type="hamming",frame_length=0.025, frame_shift=0.01,
                              cepstral_lifter=26.0,low_freq=0, vtln_low=60, vtln_high=7200, high_freq=audio.sample_rate/2)
    d_processor = DeltaPostProcessor(order=2)
    mfcc_static = processor.process(audio, vtln_warp=1.0)
    mfcc_deltas = d_processor.process(mfcc_static)
    features = np.float64(mfcc_deltas._to_dict()["data"])

    if norm=="cmvn":
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    return features


def do_html(l, p):
    l = sorted(l, key=lambda x: (x[0], x[1]))
    with open("result.html", mode="w", encoding="utf8") as ficEcr:
        ficEcr.write("<html>\n\t<body>\n In the codes, u is for utterance, q for query and e for example\n<br/>\n")
        for i in l:
            pres=((p[i[0]]["TP"]/p[i[0]]["TOT"])*100)
            ficEcr.write("\t\t{} -> cost: {}, <a href=\"./data/{}.wav\">{}</a> precision : {}/{} = {}\n<br/>\n".format
                         (i[0], round(i[1], 3), i[2],i[2], p[i[0]]["TP"], p[i[0]]["TOT"], round(pres, 2)))
        ficEcr.write("\t</body>\n</html>")

def grab_data(root, path, rep, norm, model_n, feat, lang):
    # Grab spoken lexicon and return word, crop and duration of the word
    print("opening json file")
    with io.open(root+path, mode="r",encoding="utf-8") as json_file:
        lex = json.load(json_file)
    print("json file opened")
    search_mfcc_list = []
    if rep == "wtv":
        linear_trans = make_linear(512,39)
        model_name=(model_n)
        model = Prediction(model_name, 0)
        for elt in lex:
            wav_fn = elt["crop"]
            signal, sr = read_audio(wav_fn)
            z, c = model(signal)
            if feat=='reduc':
            
                query_mfcc = c.transpose()
                query_mfcc = query_mfcc.copy(order='C')
                linear_trans = make_linear(512,39)
                query_mfcc = linear_trans(Variable(torch.from_numpy(query_mfcc)))
                query_mfcc = np.float64(query_mfcc.detach().numpy())
            else:
                if norm == "cmvn":
                    p = (c - np.mean(c, axis=0)) / np.std(c, axis=0)
                    query_mfcc = (np.float64(p)).transpose()
                    query_mfcc = query_mfcc.copy(order='C')
                else:
                    query_mfcc = (np.float64(c)).transpose()
                    query_mfcc = query_mfcc.copy(order='C')
            
            dur = len(signal) / sr * 1000
            query = {}
            query["duree"] = dur
            query["data"] = []
            query["data"].append(query_mfcc)
            query["word"] = elt[lang]
            query["ref"] = []
            query["ref"].append(elt["ref"])
            query["thres"] = 1
            search_mfcc_list.append(query)
    else :
        for elt in lex:
            wav_fn = elt["crop"]
            rate, signal = wav.read(wav_fn)
            dur = len(signal) / rate * 1000
            if rep == "vtln":
                query_mfcc = get_mfcc_vtln(wav_fn, elt["ref"], norm, lang)
            elif rep == "mfcc":
                query_mfcc = get_mfcc_dd(wav_fn, norm)
            elif rep == "plp":
                query_mfcc = get_plp(wav_fn, norm)
            else :
                print("the rep {} is unknown".format(rep))
                exit()
            query = {}
            query["duree"] = dur
            query["data"] = []
            query["data"].append(query_mfcc)
            query["word"] = elt[lang]
            query["ref"] = []
            query["ref"].append(elt["ref"])
            query["thres"] = 1
            search_mfcc_list.append(query)
    return(search_mfcc_list)


def grab_corp(root, corpus, rep, norm, model_n, feat, lang):
    # compare the queries to the utterances and return a dico with dtw value
    audios = []
    # Grabbing spoken utterances
    if os.path.isfile("./mfcc_corp_{}_{}_{}.pkl".format(rep, norm,feat)):
        print("Reading corpus mfccs")
        with open("./mfcc_corp_{}_{}_{}.pkl".format(rep, norm, feat), mode='rb') as jfile:
            audios = pickle.load(jfile)
    else:
        if rep == 'wtv':
            model_name=(model_n)
            model = Prediction(model_name, 0)
        for wav_fn in glob.glob(path.join(root + corpus, "*.wav")):
            print("Reading:", wav_fn)
            dic = {}
            dic["file"] = wav_fn
            if rep == "vtln":
                dic["data"] = get_mfcc_vtln(wav_fn, wav_fn, norm, lang)
            elif rep == "mfcc":
                dic["data"] = get_mfcc_dd(wav_fn, norm)
            elif rep == "plp":
                dic["data"] = get_plp(wav_fn, norm)
            elif rep == "wtv":
                signal, sr = read_audio(wav_fn)
                z, c = model(signal)
                if feat == 'reduc':
                
                    dic['data'] = c.transpose()
                    dic["data"] = dic["data"].copy(order='C')
                    linear_trans=make_linear(512,39)
                    dic["data"] = linear_trans(Variable(torch.from_numpy(dic["data"])))
                    dic["data"] = np.float64(dic["data"].detach().numpy())
                else:
                    if norm =="cmvn":
                        p= (c - np.mean(c, axis=0)) / np.std(c, axis=0)
                        dic["data"] = (np.float64(p)).transpose()
                        dic["data"] = dic["data"].copy(order='C')
                    else:
                        dic["data"] = (np.float64(c)).transpose()
                        dic["data"] = dic["data"].copy(order='C')
            else:
                print("the norm {} is unknown".format(rep))
                exit()
            audios.append(dic)
        with io.open("./mfcc_corp_{}_{}_{}.pkl".format(rep, norm, feat), mode='wb') as corp_json_file:
            pickle.dump(audios, corp_json_file)
    return audios

def crop_audio(elt, rep, norm, model, feat, lang):
    # time = elt["whole"].index(elt["cost"]) * 30
    time = elt["time"]*1000
    deb = time - 0.5
    if not os.path.isdir("./data_{}/".format(feat)):
        os.mkdir("./data_{}/".format(feat))
    if deb<0:
        deb=0
    fin = time + elt["duree"] +1
    ext = AudioSegment.from_wav(elt['ref'])
    ext = ext[deb:fin]
    ext.export("./data_{}/{}.wav".format(feat, elt["code"]), format("wav"))
    if rep == "vtln":
        sortie = get_mfcc_vtln("./data_{}/{}.wav".format(feat,elt["code"]), elt["ref"], norm, lang)
    elif rep == "mfcc":
        sortie=get_mfcc_dd("./data_{}/{}.wav".format(feat,elt["code"]), norm)
    elif rep == "plp":
        sortie = get_plp("./data_{}/{}.wav".format(feat,elt["code"]), norm)
    elif rep == 'wtv':
        signal, sr = read_audio("./data_{}/{}.wav".format(feat,elt["code"]))
        z, c = model(signal)
        if feat=='reduc':
            sortie = c.transpose()
            sortie = sortie.copy(order='C')
            linear_trans = make_linear(512,39)
            sortie = linear_trans(Variable(torch.from_numpy(sortie)))
            sortie = np.float64(sortie.detach().numpy())
        else:
            if norm =="cmvn":
                p = (c - np.mean(c, axis=0)) / np.std(c, axis=0)
                sortie = (np.float64(p)).transpose()
                sortie = sortie.copy(order='C')
            else:
                sortie = (np.float64(c)).transpose()
                sortie = sortie.copy(order='C')
    return sortie



class DTW():
    def __init__(self, queries, search, norm, rep, model_n, limit, feat, lang, wind):
        self.limit = limit
        if self.limit:
            self.mode="thr"
        else:
            self.mode="no_thr"
        self.rep = rep
        self.norm = norm
        if self.rep == "wtv":
            model_name=(model_n)
            self.model = Prediction(model_name, 0)
        else:
            self.model=None
        self.lang=lang
        self.wind = wind
        self.feat=feat
        self.save_html=[]
        self.queries = queries
        self.search = search
        self.checked = set()
        # if os.path.isfile("./dtw_scores_{}.pkl".format(self.norm)):
        #     with open("./processed/dtw_scores_{}.pkl".format(self.norm), mode="rb") as jfile:
        #         self.dtw_costs=pickle.load(jfile)
        # else:
        self.dtw_costs = {}
        #     print("no dtw_scores already computed")
        # if os.path.isfile("./processed/c_dict_{}.pkl".format(self.norm)):
        #     with open("./processed/c_dict_{}.pkl".format(self.norm), mode="rb") as jfile:
        #         self.c_dict = pickle.load(jfile)
        # else:
        #     print("no c_dict already computed")
        self.c_dict = set()
        self.threshold = 1
        # if os.path.isfile(".processed/pres_mot_{}.pkl".format(self.norm)):
        #     with open("./processed/pres_mot_{}.pkl".format(self.norm), mode="rb") as p:
        #         self.pres_mot = pickle.load(p)
        # else:
        self.pres_mot = {}
        # if os.path.isfile("./processed/pres_{}.pkl".format(self.norm)):
        #     with open("./processed/pres_{}.pkl".format(self.norm), mode="rb") as p:
        #         self.precision = pickle.load(p)
        #     with open("./processed/recall_{}.pkl".format(self.norm)) as p:
        #         self.recall = pickle.load(p)
        # else:
        self.precision = {}
        self.precision["TP"] = 0
        self.precision["TOT"] = 0
        self.recall = 0
        # if os.path.isfile("./processed/parite_{}.pkl".format(self.norm)):
        #     with open("./processed/parite_{}.pkl".format(self.norm), mode="rb") as p:
        #         self.par_ite = pickle.load(p)
        # else:
        self.par_ite = {}
    def get_dtw(self):
        return self.dtw_costs


    def eval_lex(self, corp, size_queries, cur):
        found=0
        tot=0
        deb = cur*20
        queries = self.queries#[deb:size_queries]
        if self.lang == 'mboshi':
            rep = '*.mb.cleaned'
        else:
            rep = '*.txt'

        for query in range(deb, size_queries):
            mot = queries[query]["word"]
            ref = queries[query]["ref"][0]
            ind_quer = query

            for utt in self.search:
                if os.path.basename(utt["file"]) == ref:
                    ind_utt = self.search.index(utt)
            code=(ind_quer, ind_utt, 0)

            if code not in self.c_dict:
                self.precision["TP"] = self.precision["TP"] + 1
                self.precision["TOT"] = self.precision["TOT"] + 1
                for gold in sorted(glob.glob(path.join(corp, rep))):
                    with open(gold, mode="r", encoding="utf8") as g_op:
                        if mot in g_op.read().split():
                            self.recall+=1

            self.c_dict.add(code)
        # with open("./processed/recall_{}.pkl".format(self.norm), mode="wb") as p:
        #     pickle.dump(self.recall, p)
        return tot

    def do_dtw(self, size_queries, ite, cur):
        """

        :param size_queries:
        :return:
        """
        print(datetime.now())
        print(len(self.search))
        bar = Bar("Processing dtw", max=len(self.search))
        for elt in range(0,len(self.search)):
            for query in range(0,len(self.queries[0:size_queries])):
                for inst in range(0,len(self.queries[query]["data"])):
                    code = (query, elt, inst)
                    if code not in self.c_dict:
                        ref = self.search[elt]["file"]
                        mot = self.queries[query]["word"]
                        query_mfcc = self.queries[query]["data"][inst]
                        search_mfcc = np.asarray(self.search[elt]["data"])
                        costs = qbe.dtw_sweep(query_mfcc, search_mfcc, self.wind)
                        self.c_dict.add(code)
                        if mot not in self.dtw_costs:
                            self.dtw_costs[mot] = []
                        fin = {}
                        fin["duree"] = self.queries[query]["duree"]
                        fin["word"] = mot
                        fin["ref"] = ref
                        # fin["costs"] = costs
                        fin["cost"] = np.min(costs)
                        fin['ch_code'] = (query,elt)
                        fin['time'] = costs.index(fin["cost"])*3/100
                        fin["code"] = "u{}q{}e{}".format(elt, query, inst)
                        fin["checked"] = False
                        self.dtw_costs[mot].append(fin)
            bar.next()
        bar.finish()

        # with open("./processed/dtw_scores_{}.pkl".format(self.norm), mode="wb") as jfile:
        #     pickle.dump(self.dtw_costs, jfile)
        # with open("./processed/c_dict_{}.pkl".format(self.norm), mode="wb") as jfile:
        #     pickle.dump(self.c_dict, jfile)
        print("dtw computed")
        self.par_mot=True
        print(datetime.now())

    def do_precision(self, quota, max_inst, aligned_fold):
        """

        :param quota:
        :return:
        """
        ######Ordering results###########
        TP = []
        print("computing precision")
        for mot in self.dtw_costs:
            self.dtw_costs[mot] = [x for x in self.dtw_costs[mot] if x["cost"] < self.threshold]
        for mot in self.dtw_costs:
            self.dtw_costs[mot] = sorted(self.dtw_costs[mot], key=lambda x: (x["cost"]))
        found = 0
        cost_temp = 0
        cpt_quota = 0
        
        verif = {}#dict to check if each word is checked
        #pour chaque dtw score
        ########################################

        for mot in sorted(self.dtw_costs, key=lambda mot : len(self.dtw_costs[mot])):
            verif[mot] = 0
            found_m = 0
            #I check each word if it has not been checked according to q_mot and quota
            for elt in self.dtw_costs[mot]:
                # if quota > 0 :
                if verif[mot]<quota:
                    if (elt["checked"] == False) and (elt['ch_code'] not in self.checked):
                        verif[mot]+=1
                        cpt_quota +=1
                        elt["checked"] = True
                        if self.lang == "mboshi":
                            file_ref = aligned_fold+(os.path.basename(elt["ref"])).replace(".wav", ".wrd")
                            # file_ref = elt["ref"].replace(".wav", ".mb.cleaned")
                            with open(file_ref, mode='r', encoding='utf-8') as op_ref:
                                gold = op_ref.read().lower()

                            if elt["word"] in gold:
                                wrd=gold.split("\n")
                                for line in wrd:
                                    linesp=line.split()
                                    if len(linesp)>2:

                                        if (elt["time"]>float(linesp[0])-0.5 and elt["time"]<float(linesp[0])+0.5)and (elt['ch_code'] not in self.checked):
                                            if elt["word"]==linesp[2].lower():
                                                self.checked.add(elt['ch_code'])
                                                found += 1
                                                found_m +=1
                                                TP.append(elt["word"])
                                                if elt["cost"] > cost_temp:
                                                    cost_temp = elt["cost"]
                                                for inp in self.queries:
                                                    if (elt['word'] in inp["word"]) and (elt['ref'] not in inp['ref']):
                                                        if len(inp['ref']) < max_inst:
                                                            inst = crop_audio(elt, self.rep, self.norm, self.model, self.feat, self.lang)
                                                            inp['data'].append(inst)
                                                            inp['ref'].append(elt['ref'])
                                                            self.save_html.append([elt["word"], elt["cost"], elt["code"]])
                            # else:
                            #     if not os.path.isdir('./FalsePos_{}'.format(elt['word'])):
                            #         os.mkdir('./FalsePos_{}'.format(elt['word']))
                            #     time = elt["time"]*1000
                            #     deb = time - 0.5
                            #     if deb<0:
                            #         deb=0
                            #     fin = time + elt["duree"] +1
                            #     ext = AudioSegment.from_wav(elt['ref'])
                            #     ext = ext[deb:fin]
                            #     ext.export("/home/getalp/leferrae/thesis/speech_dtw/FalsePos_{}/{}.wav".format(elt['word'], elt['code']), format("wav"))
                        elif self.lang == 'kun':
                            file_ref = aligned_fold+(os.path.basename(elt["ref"])).replace(".wav", ".TextGrid")
                            tg = tgio.openTextgrid(file_ref)
                            tier=tg.tierNameList[0]
                            for tg_part in (tg.tierDict[tier].entryList):
                                if (elt["time"]>tg_part[0]-0.1 and elt["time"]<tg_part[0]+0.1) and (elt['ch_code'] not in self.checked):
                                    if elt["word"]==tg_part[2].lower():
                                        self.checked.add(elt['ch_code'])
                                        found += 1
                                        found_m +=1
                                        TP.append(elt["word"])
                                        csn=False
                                        if elt["cost"] > cost_temp:
                                            cost_temp = elt["cost"]
                                        for inp in self.queries:
                                            if (elt['word'] in inp["word"]) and (elt['ref'] not in inp['ref']):
                                                if len(inp['ref']) < max_inst:
                                                    inst = crop_audio(elt, self.rep, self.norm, self.model, self.feat, self.lang)
                                                    inp['data'].append(inst)
                                                    inp['ref'].append(elt['ref'])
                                                    self.save_html.append([elt["word"], elt["cost"], elt["code"]])
                                
            if mot not in self.pres_mot:
                self.pres_mot[mot] = {}
                self.pres_mot[mot]["TP"] = found_m
                self.pres_mot[mot]["TOT"] = verif[mot]
            else:
                self.pres_mot[mot]["TP"] += found_m
                self.pres_mot[mot]["TOT"] += verif[mot]
        # with open("./processed/pres_mot_{}.pkl".format(self.norm), mode="wb") as p:
        #     pickle.dump(self.pres_mot, p)
        if self.limit:
            if self.threshold == 1:
                self.threshold = cost_temp
        if cpt_quota != 0:
            print("precision based on the iteration")
            print((found / cpt_quota) * 100)
            print("{}/{}".format(found, cpt_quota))
        self.precision["TP"] = self.precision["TP"] + found
        self.precision["TOT"] = self.precision["TOT"] + found #the False Positive are corrected, only good example added

        ###save checkpoint ####
        # with open("./processed/pres_mot_{}.pkl".format(self.norm), mode="wb") as p:
        #     pickle.dump( self.pres_mot, p)
        # with open("./processed/pres_{}.pkl".format(self.norm), mode="wb") as p:
        #     pickle.dump( self.precision, p)
        print("retrieved by iteration : ")
        f_par_ite = {}
        for i in TP:
            for ind in range(0, 5):
                if ind in self.par_ite:
                    if i in self.par_ite[ind]:
                        if ind not in f_par_ite:
                            f_par_ite[ind] = 0
                        f_par_ite[ind] += 1
        for i in f_par_ite:
            print("{} found from iteration {}".format(f_par_ite[i], i))
        # with open("./processed/parite_{}.pkl".format(self.norm), mode="wb") as p:
        #     pickle.dump(self.par_ite, p)

    def eval(self, size_queries, corp, root, cur):

        self.eval_lex(corp, size_queries, cur)
        tot = self.recall
        print("recall : {}  {}/{}\nprecision : {}  {}/{}".format((self.precision["TP"] / tot) * 100,
                                                           self.precision["TP"], tot,
                                                           (self.precision["TP"]/self.precision["TOT"]*100),
                                                                 self.precision["TP"], self.precision["TOT"]))


        for mot in self.pres_mot:
            if self.pres_mot[mot]["TOT"]==0:
                print("precision {} = {}".format(mot, 0).encode('utf-8'))
            else:
                print("precision {} = {}".format(mot, (self.pres_mot[mot]["TP"]/self.pres_mot[mot]["TOT"])*100).encode('utf-8'))
        # with open("eval1010.txt", mode="w", encoding="utf8") as ficEcr:
        #     ficEcr.write("recall : {}\nprecision : {}\n".format((self.precision["TOT"] / self.recall) * 100, self.precision["TP"]/self.precision["TOT"]))
        #     for mot in self.pres_mot:
        #         if self.pres_mot[mot]["TOT"]==0:
        #             ficEcr.write(
        #                 "precision {} = {}\n".format(mot, 0))
        #         else:
        #             ficEcr.write("precision {} = {}\n".format(mot, (self.pres_mot[mot]["TP"] / self.pres_mot[mot]["TOT"]) * 100))
        # do_html(self.save_html, self.pres_mot)


    def recap_queries(self, cur, size_queries):
        self.par_ite[cur] = []
        deb = cur * 20
        for i in self.queries[deb:size_queries]:
            self.par_ite[cur].append(i["word"])
