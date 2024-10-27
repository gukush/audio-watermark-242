from watermarks import SimpleWatermark
import os
from pydub import AudioSegment
from metrics import CorrelationMetric
import argparse
import logging
from utils import create_watermark
import numpy as np
import soundfile as sf
import torch
import time
#first empty line is used to indicate lack of watermark (no modifcation)
SUPPORTED_WATERMARKS = ['','audioseal','wavmark','silentcipher']
SUPPORTED_VOICE_CLONING = ['','openvoice']
SUPPORTED_AUDIO_EXTENSIONS = ['.wav','.opus']
ROOT_DIR = os.path.dirname(__file__)

def main_old():
    audio_dir = 'audio'
    original_audio_path = os.path.join(audio_dir,'audio1.mp3')
    watermark_path = os.path.join(audio_dir,'watermark1.mp3')
    watermarked_audio_path = os.path.join(audio_dir,'watermarked_audio.wav')
    watermark = SimpleWatermark(watermark_path)
    audio = AudioSegment.from_file(original_audio_path)

    watermarked_audio = watermark.add_watermark(audio)
    # Evaluate the quality of the watermarked audio
    correlation_metric = CorrelationMetric()
    quality = correlation_metric.evaluate_quality(audio,watermarked_audio)#evaluate_quality(original_audio, watermarked_audio)
    print(f"Quality of the watermarked audio (correlation coefficient): {quality}")

def watermark_samples(samples,watermark_list):
    watermarks = []
    for watermark in watermark_list:
        watermarks.append(create_watermark(watermark))
    for sample in samples:
        for watermark in watermarks:
            filename = os.path.basename(sample)
            logging.info(f"Processing watermark {watermark.name} for sample {filename}")
            start = time.time()
            with torch.no_grad():
                watermarked_sample, sr = watermark.add_watermark(sample)
            end = time.time()
            duration = end - start
            logging.info(f"Ended processing watermark {watermark.name} for sample {filename}, time: {duration}")
            sample_name, ext = os.path.splitext(filename)
            watermarked_path = os.path.join(ROOT_DIR,'audio','watermarked',f'{sample_name}_{watermark.name}{ext}')
            sf.write(watermarked_path,watermarked_sample,sr)


def main(args):
    os.makedirs(os.path.join(ROOT_DIR,'audio','watermarked'),exist_ok=True)
    print(args)
    samples = parse_samples(args.samples)
    watermarks = parse_technique_list(args.watermarks,SUPPORTED_WATERMARKS)
    watermark_samples(samples,watermarks)
    print(watermarks)
    pass

def parse_technique_list(str,supported_list):
    if str == 'all':
        return supported_list[1:] # skip empty string
    used_techniques = []
    names = str.split(',')
    for name in names:
        if '-' in name: # treat it as range of values
            values = name.split('-')
            assert len(values) == 2, "Incorrectly defined range in options"
            assert values[0].isdigit(), "Wrong number for range of values"
            assert values[1].isdigit(), "Wrong number for range of values"
            lower = int(values[0])
            upper = int(values[1])
            used_techniques.extend(supported_list[lower:(upper+1)])
        elif name.isdigit():
            used_techniques.extend([name])
        else:
            # treat it as name of technique
            if name.lower() in supported_list:
                used_techniques.extend([name.lower()])
    used_techniques = list(dict.fromkeys(used_techniques))
    return used_techniques

def parse_samples(samples_list):
    file_list = []
    for sample in samples_list:
        if os.path.isdir(sample):
            for item in os.listdir(sample):
                file_path = os.path.join(sample,item)
                if os.path.isfile(file_path):
                    extension = os.path.splitext(file_path)[1]
                    if extension in SUPPORTED_AUDIO_EXTENSIONS:
                        file_list.append(file_path)
                    else:
                        print(f"Warning {file_path} is not a supported audio file")
        elif os.path.isfile(sample):
            extension = os.path.splitext(file_path)[1]
            if extension in SUPPORTED_AUDIO_EXTENSIONS:
                file_list.append(file_path)
            else:
                print(f"Warning {file_path} is not a supported audio file")
    print(file_list)
    return file_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Tool for running audio watermarking and voice cloning pipeline")
    parser.add_argument("--samples",nargs='+',help="Input audio file or directory with audio files that will be used for watermarking")
    parser.add_argument("--watermarks",help="String comma-delimioted of numbers or names or simply \"all\" that indicate watermarking techniques to use in pipeline")
    parser.add_argument("--clones",help="String comma-delimioted of numbers or names or simply \"all\" that indicate voice cloning techniques to use in pipeline")
    args = parser.parse_args()
    #parser.add_argument("--distortions",help="String comma-delimioted of numbers or names or simply \"all\" that indicate watermarking techniques to use in pipeline")
    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        handlers = [
            logging.FileHandler("log"),
            logging.StreamHandler()
        ]
    )
    main(args)
