import torch
from TTS.api import TTS

tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cpu")
tts.voice_conversion_to_file(source_wav="/project/audio/voice-hispanic-1.wav", target_wav="/project/audio/voice-polish-8.wav", file_path="output.wav")