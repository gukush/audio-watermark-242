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

def download_audio(url,output):
    if not os.path.isfile(output):
        gdown.download(url=url,output=output,fuzzy=True)

def audio_watermarking():
    url = "https://drive.google.com/file/d/1Ks8Wil0ZIq5oTa7DItGmpd8bI7jXcOJM/view?usp=drive_link" #voice-hispanic-1.opus
    output = "/project/audio/voice-hispanic-1.opus"
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

def main():
    audio_watermarking()
    voice_cloning()
    #subprocess.call(['python','NISQA/run_predict.py','--mode','predict_file','--pretrained_model',
    #                 '/proj/NISQA/weights/nisqa.tar','--deg','/proj/'+output,'--output_dir','/proj/results'])


if __name__ == '__main__':
    main()