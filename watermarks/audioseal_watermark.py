from audioseal import AudioSeal
from base_watermark import BaseWatermark
#import ffmpeg
#import torch
import torchaudio

class AudiosealWatermark(BaseWatermark):
    name = "audioseal"
    def __init__(self):
        self.detector = None
        self.model = None
    def add_watermark(self, input,skip_preprocessing=False):
        """
        Input: if skip_preprocessing=False then filename, BytesIO;
            else torch Tensor of audio and sample rate (int).
        Adds audioseal watermark.
        Return a Tensor of dimensions [1,num_frames]
        """
        if self.model is None:
            self.model = AudioSeal.load_generator("audioseal_wm_16bits")
        if not skip_preprocessing:
            audio, sr = self.preprocess(input)
        else: # assuming input is tuple of (audio,sr)
            audio, sr = input
        audio = audio.unsqueeze(0) # this adds new dimension which is batch size for model
        watermark = self.model.get_watermark(audio,sample_rate=sr)
        watermarked_audio = audio + watermark
        if watermarked_audio.dim() == 3 and watermarked_audio.shape[0] == 1:
            watermarked_audio = watermarked_audio.squeeze(0)
        return watermarked_audio

    def preprocess(self, input):
        """
        Input: filename, BytesIO object
        Output: tuple of torch tensor for adding watermark from audioseal model, sample rate
        """
        audio, sr = torchaudio.load(input)
        audio_mono = audio.mean(dim=0,keepdim=True)
        return audio_mono, sr

    def detect_watermark(self, input):
        """
        Function that detects watermark.
        Input: tuple of (audio,sample rate)
        """
        audio, sr = input
        if self.detector is None:
            self.detector = AudioSeal.load_detector(("audioseal_detector_16bits"))
        result, message = self.detector.detect_watermark(audio,sample_rate=sr)
        return result, message