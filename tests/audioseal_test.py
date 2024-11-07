from audioseal import AudioSeal
import torchaudio
import os
project_root =  os.path.join(os.path.dirname(__file__),'..')
output = os.path.join(project_root,'audio','old','voice-hispanic-1.wav')
model = AudioSeal.load_generator("audioseal_wm_16bits")
audio, sr = torchaudio.load(output)
audio_mono = audio.mean(dim=0,keepdim=True)
audios = audio_mono.unsqueeze(0)
watermark = model.get_watermark(audios,sample_rate=sr)
watermarked_audio = audios + watermark
watermark_path = output.replace('.wav','-watermarked_mono.wav')
if watermarked_audio.dim() == 3 and watermarked_audio.shape[0] == 1:
    watermarked_audio = watermarked_audio.squeeze(0)
torchaudio.save(watermark_path,watermarked_audio,sr)
detector = AudioSeal.load_detector(("audioseal_detector_16bits"))
result, message = detector.detect_watermark(watermarked_audio,sample_rate=sr)
