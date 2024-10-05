import wavmark
import torch
import soundfile
import numpy as np
from base_watermark import BaseWatermark
# Not tested yet if it works properly
class WavmarkWatermark(BaseWatermark):
    name = "wavmark"
    payload = np.array([1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0])
    def __init__(self):
        self.model = None
    def add_watermark(self, input,skip_preprocessing=False):
        if self.model is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.model = wavmark.load_model().to(device)
        if not skip_preprocessing:
            audio, _ = self.preprocess(input)
        else:
            audio, _ = input
        watermarked_audio, _ = wavmark.encode_watermark(self.model,audio,self.payload,show_progress=True)
        return watermarked_audio
    def preprocess(self, input):
        audio, sr = soundfile.load(input)
        return audio, sr
    def detect_watermark(self, audio):
        if self.model is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.model = wavmark.load_model().to(device)
        payload_decoded, info = wavmark.decode_watermark(self.model,audio,show_progress=True)
        return payload_decoded, info
