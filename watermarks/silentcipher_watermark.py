import silentcipher
import librosa
from .base_watermark import BaseWatermark
import torch
from config import MAX_SIZE_SILENTCIPHER
import numpy as np

class SilentcipherWatermark(BaseWatermark):
    name = "silentcipher"
    payload = [0,0,0,0b01100001,0b00110010] # [123, 234, 111, 222, 11]
    def __init__(self):
        self.model = None
    def add_watermark(self, input,skip_preprocessing=False):
        if self.model is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = silentcipher.get_model(model_type='16k',device=device) #model_type='44.1k'
        if not skip_preprocessing:
            audio, sr = self.preprocess(input)
        else:
            audio, sr = input
        num_samples = audio.shape[0]
        print(f"num samples: {num_samples}")
        print(MAX_SIZE_SILENTCIPHER)
        if num_samples > MAX_SIZE_SILENTCIPHER:
            parts = []
            breakpoint()
            for start in range(0,num_samples,MAX_SIZE_SILENTCIPHER):
                end = min(start + MAX_SIZE_SILENTCIPHER, num_samples)
                parts.append(audio[start:end])
            processed_parts = []
            for part in parts:
                watermarked_part, sdr = self.model.encode_wav(part,sr,self.payload)
                processed_parts.append(watermarked_part)
            breakpoint()
            watermarked_audio = np.concatenate(processed_parts)
        else:
            watermarked_audio, sdr = self.model.encode_wav(audio,sr,self.payload)
        return watermarked_audio, sr
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

