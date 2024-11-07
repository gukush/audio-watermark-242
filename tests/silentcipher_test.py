
import silentcipher
import soundfile
import librosa
import os
project_root = os.path.join(os.path.dirname(__file__),'..')
device = 'cpu'
model = silentcipher.get_model(model_type='44.1k',device=device)
audio, sr = librosa.load(os.path.join(project_root,'audio/old/voice-polish-1.wav'))
encoded, sdr = model.encode_wav(audio, sr, [123, 234, 111, 222, 11])
soundfile.write(os.path.join(project_root,'audio/voice-polish-1_silentcipher.wav'),encoded,sr,format='wav')
result = model.decode_wav(encoded,sr,phase_shift_decoding=False)
print(result)