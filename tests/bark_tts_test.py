import os
import sys
# This test requires a lot of resources
root_path = os.path.join(os.path.dirname(__file__),'..')#,'bark_with_voice_clone')
sys.path.append(root_path)
submodule_path = os.path.join(root_path,'bark-with-voice-clone')
from bark_with_voice_clone.bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio

from bark_with_voice_clone.bark.api import generate_audio, semantic_to_waveform
from transformers import BertTokenizer
from bark_with_voice_clone.bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic


text_prompt = "This is test for voice generation."
voice_name = 'polish_speaker_0'
x_semantic = generate_text_semantic(
    text_prompt,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)

x_coarse_gen = generate_coarse(
    x_semantic,
    history_prompt=voice_name,
    temp=0.7,
    top_k=50,
    top_p=0.95,
)
x_fine_gen = generate_fine(
    x_coarse_gen,
    history_prompt=voice_name,
    temp=0.5,
)
audio_array = codec_decode(x_fine_gen)
audio_output_path = os.path.join(root_path,'audio','test-tts_polish-speaker-0.wav')
import soundfile
soundfile.write(audio_output_path,audio_array,sample_rate=24000)