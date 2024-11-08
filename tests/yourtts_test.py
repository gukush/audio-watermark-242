import torch
from TTS.api import TTS
import os
import subprocess
# Example voice cloning with YourTTS in English, French and Portuguese
device = 'cpu'
project_root = os.path.join(os.path.dirname(__file__),'..')
source_file = os.path.join(project_root,'audio','old','voice-hispanic-1.wav')
target_file = os.path.join(project_root,'audio','old','voice-polish-8.wav')
output_file = os.path.join(project_root,'audio','old','voice-hispanic-1_yourtts.wav')
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
#tts.voice_conversion_to_file(source_wav=source_file, target_wav=target_file, file_path=output_file)
try:
    #tts --model_name tts_models/multilingual/multi-dataset/your_tts  --speaker_wav /project/audio/old/voice-hispanic-1.wav --reference_wav  /project/audio/old/voice-polish-8.wav --language_idx "en"

    result = subprocess.run(
        ['tts','--model_name','tts_models/multilingual/multi-dataset/your_tts',
         '--speaker_wav',source_file,'--reference_wav',target_file,
         '--language_idx','en','--out_path',output_file]
    )
    print(result.stdout)
except Exception as e:
    raise e