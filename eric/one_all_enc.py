from os import path
import tensorflow as tf
from progress.bar import Bar
from python_speech_features import delta
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from tflego import NP_DTYPE, TF_DTYPE, NP_ITYPE, TF_ITYPE
import numpy as np
from batching import SimpleIterator
import tflego
import data
import os
from scipy.spatial.distance import cdist
import json

def build_cae_from_options_dict(a, a_lengths, b_lengths, options_dict):
    # Latent layer
    build_latent_func = tflego.build_autoencoder
    latent_func_kwargs = {
        "enc_n_hiddens": [],
        "n_z": options_dict["n_z"],
        "dec_n_hiddens": [options_dict["dec_n_hiddens"][0]],
        "activation": tf.nn.relu
    }

    # Network
    network_dict = tflego.build_multi_encdec_lazydynamic_latentfunc(
        a, a_lengths, options_dict["enc_n_hiddens"],
        options_dict["dec_n_hiddens"], build_latent_func, latent_func_kwargs,
        y_lengths=b_lengths, rnn_type=options_dict["rnn_type"],
        bidirectional=options_dict["bidirectional"],
        keep_prob=options_dict["keep_prob"],
        add_conditioning_tensor=None)

    encoder_states = network_dict["encoder_states"]
    ae = network_dict["latent_layer"]
    z = ae["z"]
    y = network_dict["decoder_output"]
    mask = network_dict["mask"]
    y *= tf.expand_dims(mask, -1)  # safety

    return {"z": z, "y": y, "mask": mask}


def get_mfcc_dd(wav_fn, cmvn=True):
    """Return the MFCCs with deltas and delta-deltas for a audio file."""
    (rate, signal) = wav.read(wav_fn)
    mfcc_static = mfcc(signal, rate)
    mfcc_deltas = delta(mfcc_static, 2)
    mfcc_delta_deltas = delta(mfcc_deltas, 2)
    features = np.hstack([mfcc_static, mfcc_deltas, mfcc_delta_deltas])
    if cmvn:
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    return features


def segment(sent, step=3, wind=[100, 70, 40]):
    """

    :param corp: corpus from which the segment are extracted
    :param step:
    :param wind: the lenghts of the differents segments extracted at a same time
    :return: a stack of embeded segments and a reference file
    """
    segs = []
    seg_ref = {}
    cpt = 0
    utt_lengths = []
    i_start = 0
    w1 = []
    w07 = []
    w04 = []
    n_search = sent.shape[0]
    while i_start <= n_search - 1 or i_start == 0:
        for ind in wind:
            if ind == 100:
                w1.append(cpt)
            elif ind == 70:
                w07.append(cpt)
            elif ind == 40:
                w04.append(cpt)
            sref = {}
            sref["wind"] = ind
            sref["time"] = i_start / 100
            seg_ref[cpt] = sref
            seg = sent[i_start:i_start + ind]
            utt_lengths.append(seg.shape[0])
            i_start += step
            cpt += 1
            segs.append(seg)
    return segs, utt_lengths, w1, w04, w07


options_dict = {
    "train_tag": "utd",  # "gt", "gt2", "utd", "rnd",
    # "besgmm", "besgmm7"
    "pretrain_tag": None,  # if not provided, same tag as
    # train_tag is used
    "max_length": 100,
    "min_length": 10,  # only used with "rnd" train_tag or
    # or pretrain_tag
    "bidirectional": True,
    "rnn_type": "gru",  # "lstm", "gru", "rnn"
    "enc_n_hiddens": [400, 400, 400],
    "dec_n_hiddens": [400, 400, 400],
    "n_z": 130,  # latent dimensionality
    "learning_rate": 0.001,
    "keep_prob": 1.0,
    "ae_n_epochs": 20,  # AE pretraining options
    "ae_batch_size": 32,
    "ae_n_buckets": 3,
    "pretrain_usefinal": False,  # if True, do not use best
    # validation AE model, but rather
    # use final model
    "cae_n_epochs": 2,  # CAE training options
    "cae_batch_size": 3,
    "cae_n_buckets": 3,
    "extrinsic_usefinal": False,  # if True, during final extrinsic
    # evaluation, the final saved model
    # will be used (instead of the
    # validation best)
    "use_test_for_val": False,
    "ae_n_val_interval": 1,
    "cae_n_val_interval": 1,
    "d_speaker_embedding": None,  # if None, no speaker information
    # is used, otherwise this is the
    # embedding dimensionality
    "rnd_seed": 1,
}

a = tf.compat.v1.placeholder(TF_DTYPE, [None, None, 13])
a_lengths = tf.compat.v1.placeholder(TF_ITYPE, [None])
b = tf.compat.v1.placeholder(TF_DTYPE, [None, None, 13])
b_lengths = tf.compat.v1.placeholder(TF_ITYPE, [None])
network_dict = build_cae_from_options_dict(
    a, a_lengths, b_lengths, options_dict
)
z = network_dict["z"]
data_dir = path.join("data", "first")
temp = path.join(
    "models", path.split(data_dir)[-1] + "." +
              "utd", "4cb3a05a07", "ae.tmp.ckpt")
model = path.join(temp, "ae.tmp.ckpt")
saver = tf.train.Saver()
# name = "abiayi_2015-09-09-11-03-42_samsung-SM-T530_mdw_elicit_Dico12_5"
name = "abiayi_2015-09-09-11-03-42_samsung-SM-T530_mdw_elicit_Dico12_177"
file = "/home/leferrae/Desktop/These/mboshi/mboshi-french-parallel-corpus/full_corpus_newsplit/all/" + name + ".wav"
a_file = "/home/leferrae/Desktop/These/mboshi/wrd/" + name + ".wrd"
with open ("/home/leferrae/Desktop/These/mboshi/crops/lex100/spoken_lex100.json") as j:
    lex=json.load(j)
words_feat = []
words_length = []
for ii in range(0,20):
    feat=get_mfcc_dd(lex[ii]["crop"])
    words_feat.append(feat)
    words_length.append(feat.shape[0])
sent = get_mfcc_dd(file)


segs, utt_l, w1, w04, w07 = segment(sent)
data.trunc_and_limit_dim(x=segs, lengths=utt_l, d_frame=13,
                         max_length=options_dict["max_length"])
data.trunc_and_limit_dim(x=words_feat, lengths=words_length, d_frame=13, max_length=options_dict["max_length"])
iterator = SimpleIterator(segs, len(segs), False)
iter_mot = SimpleIterator(words_feat, 1, False)
v = []
m = []

with tf.Session() as session:
    # saver.restore(session, self.val_model_fn)

    saver.restore(session, "./models/first.utd/4cb3a05a07/ae.tmp.ckpt")

    for batch_x_padded, batch_x_lengths in iterator:
        np_x = batch_x_padded
        np_x_lengths = batch_x_lengths
        np_z = session.run(
            [z], feed_dict={a: np_x, a_lengths: np_x_lengths}
        )[0]
        v.append(np_z)
    for batch_x_padded, batch_x_lengths in iter_mot:
        np_x = batch_x_padded
        np_x_lengths = batch_x_lengths
        np_z = session.run(
            [z], feed_dict={a: np_x, a_lengths: np_x_lengths}
        )[0]
        m.append(np_z)
m_vec = np.vstack(m)
vect_segs = np.vstack(v)

matrix = cdist(vect_segs, m_vec, metric="cosine")

for cols in matrix:
    print(cols)
exit()
