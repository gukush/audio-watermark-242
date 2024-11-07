import torch
from TTS.api import TTS
import os

project_root =  os.path.join(os.path.dirname(__file__),'..')
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=True).to("cpu")
tts.voice_conversion_to_file(source_wav=os.path.join(project_root,'audio','old','voice-hispanic-1.wav'), target_wav=os.path.join(project_root,'audio','old','voice-polish-8.wav'), file_path=os.path.join(project_root,'audio','output_coquitts.wav'))