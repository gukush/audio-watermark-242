from watermarks import SimpleWatermark, AudiosealWatermark, WavmarkWatermark, SilentcipherWatermark

def create_watermark(watermark_name):
    if watermark_name == "audioseal":
        return AudiosealWatermark()
    if watermark_name == "wavmark":
        return WavmarkWatermark()
    if watermark_name == "silentcipher":
        return SilentcipherWatermark()
