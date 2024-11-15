from audioseal import AudioSeal
from .base_watermark import BaseWatermark
#import ffmpeg
import torch
import torchaudio
# Not tested yet if it works properly
class AudiosealWatermark(BaseWatermark):
    name = "audioseal"
    #tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1]])
    payload = torch.Tensor([[1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0]])
    def __init__(self):
        self.detector = None
        self.model = None
    def add_watermark(self, input,skip_preprocessing=False,stereo=False):
        """
        Input: if skip_preprocessing=False then filename, BytesIO;
            else torch Tensor of audio and sample rate (int).
        Adds audioseal watermark.
        Return a Tensor of dimensions [1,num_frames]
        """
        if self.model is None:
            self.model = AudioSeal.load_generator("audioseal_wm_16bits")
        if not skip_preprocessing:
            audio, sr = self.preprocess(input,stereo)
        else: # assuming input is tuple of (audio,sr)
            audio, sr = input
        print(audio.shape)
        # for stereo the channels are watermarked separately
        if stereo and audio.shape[0] == 2:
            left_channel = audio[0]
            right_channel = audio[1]
            left_channel = left_channel.unsqueeze(0)
            right_channel = right_channel.unsqueeze(0)
            # possible overhead due to calling model inference twice
            left_watermark = self.model.get_watermark(left_channel,sample_rate=sr,message=self.payload)
            right_watermark = self.model.get_watermark(right_channel,sample_rate=sr,message=self.payload)
            left_channel_watermarked = left_channel + left_watermark
            right_channel_watermarked = right_channel + right_watermark
            watermarked_audio = torch.stack([left_channel_watermarked,right_channel_watermarked])
            return watermarked_audio.numpy(), sr
        else:
            audio = audio.unsqueeze(0) # this adds new dimension which is batch size for model
            watermark = self.model.get_watermark(audio,sample_rate=sr,message=self.payload)
            print(watermark.shape)
            watermarked_audio = audio + watermark
            watermarked_audio = watermarked_audio.cpu().detach()
            if watermarked_audio.dim() == 3 and watermarked_audio.shape[0] == 1:
                watermarked_audio = watermarked_audio.squeeze(0)
            # Reshape to shape compatible with soundfile library (after converting to numpy array)
            if watermarked_audio.dim() == 2 and watermarked_audio.shape[0] == 1:
                watermarked_audio = watermarked_audio.view(-1, 1)
            print(watermarked_audio.shape)
            return watermarked_audio.numpy(), sr

    def preprocess(self, input,stereo=False):
        """
        Input: filename, BytesIO object
        Output: tuple of torch tensor for adding watermark from audioseal model, sample rate
        """
        audio, sr = torchaudio.load(input)
        if not stereo:
            audio_mono = audio.mean(dim=0,keepdim=True)
            return audio_mono, sr
        else:
            return audio, sr

    def detect_watermark(self, input):
        """
        Function that detects watermark.
        Input: tuple of (audio,sample rate)
        Output: result - probability, message - encoded message
        """
        audio, sr = input
        audio_tensor = torch.from_numpy(audio).to(torch.float32)
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        if self.detector is None:
            self.detector = AudioSeal.load_detector(("audioseal_detector_16bits"))
        result, message = self.detector.detect_watermark(audio_tensor,sample_rate=sr)
        return result, message