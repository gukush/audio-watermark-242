from watermarks import SimpleWatermark, AudiosealWatermark, WavmarkWatermark, SilentcipherWatermark
from voice_cloning import CoquittsCloner

def create_watermark(watermark_name):
    if watermark_name == "audioseal":
        return AudiosealWatermark()
    if watermark_name == "wavmark":
        return WavmarkWatermark()
    if watermark_name == "silentcipher":
        return SilentcipherWatermark()


def create_clone(clone_name):
    if clone_name == "coquitts":
        return CoquittsCloner()