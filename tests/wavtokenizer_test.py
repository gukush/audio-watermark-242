import os
import sys
sys.path.append('/project/WavTokenizer')
from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

device = torch.device('cpu')
config_path = "/project/models/wavtokenizer/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "/project/models/wavtokenizer/wavtokenizer_medium_speech_320_24k_v2.ckpt"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path,model_path)
wavtokenizer = wavtokenizer.to(device)

audio_path = "/project/tmp/test_clone.wav" #Elevenlabs_184339_snippet2.mp3
audio, sr = torchaudio.load(audio_path)
audio = convert_audio(audio, sr, 24000, 1)

bandwidth_id = torch.tensor([0])
audio = audio.to(device)
with torch.no_grad():
    sth_tmp, discrete_code = wavtokenizer.encode_infer(audio,bandwidth_id=bandwidth_id)


#torch.save(discrete_code,"/project/tmp/Elevenlabs_latent_wavtokenizer.pth")
features = wavtokenizer.codes_to_features(discrete_code)
bandwidth_id = torch.tensor([0])
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
audio_out_path = "/project/tmp/test_clone_wavtokenizer.mp3" #Elevenlabs_184339_snippet2
torchaudio.save(audio_out_path,audio_out,24000)