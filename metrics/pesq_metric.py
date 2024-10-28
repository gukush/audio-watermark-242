# pip install https://github.com/ludlows/python-pesq/archive/master.zip
# supposodely librosa is better for resampling than scipy
# TODO: make it into a metric class and handle cases better - this is just demonstration of producing output
from scipy.io import wavfile
from scipy.signal import resample
from pesq import pesq

sr_1, audio_1 = wavfile.read("/project/audio/old/voice-polish-1.wav")
sr_2, audio_2 = wavfile.read("/project/audio/old/voice-polish-1-silentcipher.wav")
stereo = True
audio_1_resampled = resample(audio_1,16000)
audio_2_resampled = resample(audio_2,16000)
print(audio_1_resampled.shape)
if stereo:
    audio_1_resampled_left = audio_1_resampled[:,0]
    audio_1_resampled_right = audio_1_resampled[:,1]
    print(audio_1_resampled_left.shape)
    print(pesq(16000,audio_1_resampled_left,audio_2_resampled,'wb'))
