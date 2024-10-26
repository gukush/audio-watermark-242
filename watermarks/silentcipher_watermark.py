import silentcipher
import librosa
import soundfile
from base_watermark import BaseWatermark
import torch

class SilentcipherWatermark(BaseWatermark):
    name = "silentcipher"
    payload = [123, 234, 111, 222, 11]
    def __init__(self):
        self.model = None
    def add_watermark(self, input,skip_preprocessing=False):
        if self.model is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = silentcipher.get_model(model_type='44.1k',device=device)
        if not skip_preprocessing:
            audio, sr = self.preprocess(input)
        else:
            audio, sr = input
        watermarked_audio, sdr = self.model.encode_wav(audio,sr,self.payload)
        return watermarked_audio
    def preprocess(self, input):
        audio, sr = librosa.load(input)
        return audio, sr
    def detect_watermark(self, input):
        if self.model is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = silentcipher.get_model(model_type='44.1k',device=device)
        audio, sr = input
        result = self.model.decode_wav(audio,sr,phase_shift_decoding=False)
        return result

