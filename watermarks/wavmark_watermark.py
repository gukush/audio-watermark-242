import wavmark
from base_watermark import BaseWatermark

class WavmarkWatermark(BaseWatermark):
    name = "wavmark"
    def __init__(self):
        pass
    def add_watermark(self, audio):
        return super().add_watermark(audio)
    def preprocess(self, input):
        return super().preprocess(input)
    def detect_watermark(self, audio):
        return super().detect_watermark(audio)
