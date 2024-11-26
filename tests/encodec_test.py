from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch

# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz()
# The number of codebooks used will be determined bythe bandwidth selected.
# E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
model.set_target_bandwidth(12.0)

path = "/project/tmp/Elevenlabs_184339_snippet3.mp3"
path = "/project/tmp/test_clone.wav"
# Load and pre-process the audio waveform
wav, sr = torchaudio.load(path) # Elevenlabs_184339_snippet_64k.mp3
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
with torch.no_grad():
    audio_values = model.decode(encoded_frames)
#torchaudio.save(path.replace('.mp3','_encodec.mp3'),audio_values.squeeze(0),sample_rate=24000)
torchaudio.save(path.replace('.wav','_encodec.wav'),audio_values.squeeze(0),sample_rate=24000)

