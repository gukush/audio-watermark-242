import torch
from TTS.api import TTS

# Example voice cloning with YourTTS in English, French and Portuguese
device = 'cpu'
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
tts.voice_conversion_to_file(ource_wav="/audio/voice-hispanic-1.wav", target_wav="/audio/voice-polish-8.wav", file_path="output.wav")
