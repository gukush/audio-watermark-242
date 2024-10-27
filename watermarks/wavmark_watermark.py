import wavmark
import torch
import soundfile
import numpy as np
from .base_watermark import BaseWatermark
import librosa
# Not tested yet if it works properly
class WavmarkWatermark(BaseWatermark):
    name = "wavmark"
    payload = np.array([1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0])
    def __init__(self):
        self.model = None
    # Add Stereo support (watermarking channels separately (maybe in batches?))
    def add_watermark(self, input,skip_preprocessing=False):
        if self.model is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.model = wavmark.load_model().to(device)
        if not skip_preprocessing:
            audio, sr = self.preprocess(input)
        else:
            audio, sr = input
        print(audio.shape)
        watermarked_audio, sr = wavmark.encode_watermark(self.model,audio,self.payload,show_progress=True)
        return watermarked_audio, sr
    def preprocess(self, input,stereo=False):
        audio, sr = soundfile.read(input)
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        #print(audio.shape)
        #if not stereo:
        #    audio_mono = audio.mean(axis=1,keepdims=True)
        #    return audio_mono, sr
        #else:
        return audio_resampled, 16000
    def detect_watermark(self, audio):
        if self.model is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.model = wavmark.load_model().to(device)
        payload_decoded, info = wavmark.decode_watermark(self.model,audio,show_progress=True)
        return payload_decoded, info
