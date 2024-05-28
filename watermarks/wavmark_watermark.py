from wavmark.utils import wm_add_util

from .base_watermark import BaseWatermark
import torch
import wavmark
import soundfile
import numpy as np


class WavMarkWatermark(BaseWatermark):
    def __init__(self, watermark_file: str, interval=16000):
        self.watermark = np.unpackbits(np.frombuffer(open(watermark_file, 'rb').read(), dtype=np.uint8))
        assert len(self.watermark) == 32, 'Watermark size has to be 32 bits!'

        self.interval = interval
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = wavmark.load_model().to(self.device)

    def add_watermark(self, audio):
        signal, sample_rate = soundfile.read(audio)
        trunk = signal[0:self.interval]
        message_np = self.watermark

        with torch.no_grad():
            signal_wmd, info = wm_add_util.add_watermark(self.watermark, signal, 16000, 0.1,
                                                         self.device, self.model, 20, 38, show_progress=False)
            soundfile.write('./audio/watermarked.wav', signal_wmd, samplerate=16000)

        return signal_wmd
