# This file:
# - downloads audio file from url
# - adds audioseal watermark to it
# - does runs NISQA metric on it and the initial sample
import gdown
import os
import ffmpeg
from audioseal import AudioSeal
import torch
import torchaudio
import io
from scipy.io.wavfile import read as wav_read
import subprocess
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import silentcipher
import librosa
import soundfile
import psutil
import gc
#import torch.profiler as profiler
import tracemalloc
# adapted from following collab notebook: https://colab.research.google.com/github/facebookresearch/audioseal/blob/master/examples/colab.ipynb#scrollTo=007c48cb
# and this jupyter notebook: https://github.com/myshell-ai/OpenVoice/blob/main/demo_part3.ipynb

def convert_to_mp3(input_file, output_file):
    # Ensure the output format is MP3
    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(stream, output_file, format='mp3', acodec='libmp3lame')
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream)

def convert_to_wav(input_file, output_file):
    # Define the input and output process, ensuring the output is in WAV format
    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(stream, output_file, format='wav')
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream)

def voice_cloning():
    url = "https://drive.google.com/file/d/1xykjCne5zl4mlVwZ2gqU31SGIt85DZDm/view?usp=drive_link"
    output = "/project/audio/voice-polish-1.opus"
    download_audio(url,output)
    ckpt_converter = '/models/checkpoints_v2/converter'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = 'results'
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    convert_to_mp3('/project/audio/voice-hispanic-1-watermarked_mono.opus','/project/audio/voice-hispanic-1-watermarked_mono.mp3')
    reference_speaker = 'audio/voice-hispanic-1-watermarked_mono.mp3'
    target_se, audio_name = se_extractor.get_se(reference_speaker,tone_color_converter,vad=False)
    convert_to_wav('audio/voice-polish-1.opus','audio/voice-polish-1.wav')
    src_path = 'audio/voice-polish-1.wav'
    source_se, audio_name_target = se_extractor.get_se(src_path,tone_color_converter,vad=False)
    encode_message = "242"
    tone_color_converter.convert(audio_src_path = src_path,
                                 src_se = source_se,
                                 tgt_se=target_se,
                                 output_path='audio/voice-hispanic-converted.wav',
                                 message=encode_message) #they use wavmark for watermarking
    del tone_color_converter
    gc.collect()

def download_audio(url,output):
    if not os.path.isfile(output):
        gdown.download(url=url,output=output,fuzzy=True)

def audio_watermarking():
    url = "https://drive.google.com/file/d/1Ks8Wil0ZIq5oTa7DItGmpd8bI7jXcOJM/view?usp=drive_link" #voice-hispanic-1.opus
    output = "/project/audio/old/voice-hispanic-1.opus"
    download_audio(url,output)
    model = AudioSeal.load_generator("audioseal_wm_16bits")
    #recorded, sr = get_audio_from_path(output)
    audio, sr = torchaudio.load(output)
    audio_mono = audio.mean(dim=0,keepdim=True)
    audios = audio_mono.unsqueeze(0)
    watermark = model.get_watermark(audios,sample_rate=sr)
    watermarked_audio = audios + watermark
    watermark_path = output.replace('.opus','-watermarked_mono.opus')
    # Ensure watermarked_audio is 2D
    if watermarked_audio.dim() == 3 and watermarked_audio.shape[0] == 1:
        watermarked_audio = watermarked_audio.squeeze(0)
    torchaudio.save(watermark_path,watermarked_audio,sr)
    # TODO: handle paths in a better fashion
    subprocess.call(['python','NISQA/run_predict.py','--mode','predict_file','--pretrained_model',
                     '/project/NISQA/weights/nisqa.tar','--deg',watermark_path,'--output_dir','/project/results'])
    del model
    gc.collect()

def detecting_from_clone():
    input_path = "/project/audio/voice-hispanic-1-watermarked_mono.opus"
    audio, sr = torchaudio.load(input_path)
    audio = audio.unsqueeze(0)
    detector = AudioSeal.load_detector(("audioseal_detector_16bits"))
    result, message = detector.detect_watermark(audio,sample_rate=sr,message_threshold=0.5)
    print(f"\n{input_path} audio has {result*100}% probability of being watermarked")
    print(message)
    input_path = "/project/audio/voice-hispanic-converted.wav"
    audio, sr = torchaudio.load(input_path)
    audio = audio.unsqueeze(0)
    detector = AudioSeal.load_detector(("audioseal_detector_16bits"))
    result, message = detector.detect_watermark(audio,sample_rate=sr,message_threshold=0.5)
    print(f"\n{input_path} audio has {result*100}% probability of being watermarked")
    print(message)
    del detector
    gc.collect()


top_stats = None
def test_silentcipher():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = silentcipher.get_model(model_type='44.1k',device=device)
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    if device == 'cpu':
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    else:
        gpu_idx = torch.cuda.current_device()
        print(f"Total Memory: {torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 2):.2f} MB")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(gpu_idx) / (1024 ** 2):.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(gpu_idx) / (1024 ** 2):.2f} MB")
    audio, sr = librosa.load('/project/audio/old/voice-polish-1.wav')
    encoded, sdr = model.encode_wav(audio, sr, [123, 234, 111, 222, 11])
    soundfile.write("/project/audio/voice-polish-1-silentcipher.wav",encoded,sr,format='wav')
    result = model.decode_wav(encoded,sr,phase_shift_decoding=False)
    print(result)
    del model
    gc.collect()

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cpu_memory_usage", row_limit=-1))
    prof.export_chrome_trace("/project/tmp/test_trace_" + str(prof.step_num) + ".json")

def main():
    tracemalloc.start()
    with torch.no_grad():
        #with profiler.profile(
        #    activities=[torch.profiler.ProfilerActivity.CPU],
        #    with_stack=True,profile_memory=True) as prof:
        test_silentcipher()
        snapshot = tracemalloc.take_snapshot()
        #audio_watermarking()
        #snapshot = tracemalloc.take_snapshot()
        #voice_cloning()
        #snapshot = tracemalloc.take_snapshot()
        #detecting_from_clone()
        snapshot = tracemalloc.take_snapshot()
    #print(prof.key_averages().table(sort_by="self_cpu_memory_usage",row_limit=-1))
    #subprocess.call(['python','NISQA/run_predict.py','--mode','predict_file','--pretrained_model',
    #                 '/proj/NISQA/weights/nisqa.tar','--deg','/proj/'+output,'--output_dir','/proj/results'])


if __name__ == '__main__':
    main()
    print(top_stats)