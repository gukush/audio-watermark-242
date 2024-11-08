import os
import sys

submodule_path = os.path.join(os.path.dirname(__file__),'..')#,'bark_with_voice_clone')
sys.path.append(submodule_path)

from bark_with_voice_clone.bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio

device = 'cpu'
model = load_codec_model(use_gpu=False)

import torchaudio
import torch

from bark_with_voice_clone.hubert.hubert_manager import HuBERTManager

hubert_manager = HuBERTManager()
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()

from bark_with_voice_clone.hubert.pre_kmeans_hubert import CustomHubert
from bark_with_voice_clone.hubert.customtokenizer import CustomTokenizer

hubert_model = CustomHubert(checkpoint_path='/project/data/models/hubert/hubert.pt').to(device)
tokenizer = CustomTokenizer(checkpoint_path='/project/data/models/hubert/hubert.pt').to(device)

audio_filepath = '/project/audio/old/voice-polish-1.wav'
audio, sr = torchaudio.load(audio_filepath)
wav = convert_audio(audio,sr,model.sample_rate, model.channels)
wav = wav.to(device)

semantic_vectors = hubert_model.forward(wav,input_sample_hz=model.sample_rate)
semantic_tokens = tokenizer.get_token(semantic_vectors)

with torch.no_grad():
    encoded_frames = model.encode(wav.unsqueeze(0))
codes = torch.cat([encoded[0] for encoded in encoded_frames],dim=-1).squeeze()
codes = codes.cpu().numpy()

semantic_tokens = semantic_tokens.cpu().numpy()
import numpy as np
voice_name = 'output'

output_path = submodule_path+'bark/assets/prompts/' + voice_name + '.npz'
np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
converted_path = '/project/audio/old/voice_cloned_bark.wav'

from bark_with_voice_clone.bark.api import generate_audio, semantic_to_waveform
from transformers import BertTokenizer
from bark_with_voice_clone.bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic


trasnfer_voice_filepath = '/project/audio/old/voice-hispanic-1.wav'
audio_2, sr_2 = torchaudio.load(audio_filepath)
wav_2 = convert_audio(audio_2,sr_2,model.sample_rate, model.channels)
wav_2 = wav_2.to(device)

semantic_vectors_2 = hubert_model.forward(wav_2,input_sample_hz=model.sample_rate)
semantic_tokens_2 = tokenizer.get_token(semantic_vectors_2)


cloned_audio = semantic_to_waveform(semantic_tokens_2,history_prompt='output')
