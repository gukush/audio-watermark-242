# This file:
# - downloads audio file from url
# - adds audioseal watermark to it
# - does runs NISQA metric on it and the initial sample
import gdown
import os
#import ffmpeg
from audioseal import AudioSeal
import torch
import torchaudio
import io
from scipy.io.wavfile import read as wav_read
import subprocess

# adapted from following collab notebook: https://colab.research.google.com/github/facebookresearch/audioseal/blob/master/examples/colab.ipynb#scrollTo=007c48cb

def main():
    url = "https://drive.google.com/file/d/1Ks8Wil0ZIq5oTa7DItGmpd8bI7jXcOJM/view?usp=drive_link"
    output = "./audio/voice-hispanic-1.opus"
    if not os.path.isfile(output):
        gdown.download(url=url,output=output,fuzzy=True)
    model = AudioSeal.load_generator("audioseal_wm_16bits")
    #recorded, sr = get_audio_from_path(output)
    audio, sr = torchaudio.load(output)
    audio_mono = audio.mean(dim=0,keepdim=True)
    audios = audio_mono.unsqueeze(0)
    watermark = model.get_watermark(audios,sample_rate=sr)
    watermarked_audio = audios + watermark
    watermark_path = output.replace('.opus','-watermarked_mono.opus')
    torchaudio.save(watermark_path,watermarked_audio,sr)
    # TODO: handle paths in a better fashion
    subprocess.call(['python','NISQA/run_predict.py','--mode','predict_file','--pretrained_model',
                     '/project/NISQA/weights/nisqa.tar','--deg','/project/'+watermark_path,'--output_dir','/proj/results'])
    #subprocess.call(['python','NISQA/run_predict.py','--mode','predict_file','--pretrained_model',
    #                 '/proj/NISQA/weights/nisqa.tar','--deg','/proj/'+output,'--output_dir','/proj/results'])


if __name__ == '__main__':
    main()