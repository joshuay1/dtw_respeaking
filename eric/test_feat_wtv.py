import torch
from torch import nn
from fairseq.models.wav2vec import Wav2VecModel
import soundfile as sf
#from fairseq.scripts.wav2vec_featurize import Prediction

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

def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """

    wav, sr = sf.read(fname)
    assert sr == 16e3

    return wav, 16e3

model_name=('./mb_model/checkpoint_best.pt')
model = Prediction(model_name, 0)
name = "/home/getalp/leferrae/thesis/corpora/audios/martial_2015-09-07-15-24-49_samsung-SM-T530_mdw_elicit_Dico19_4.wav"
wav, sr = read_audio(name)
z, c = model(wav)
feat = c
print(feat.shape)