from watermarks import SimpleWatermark
import os
from pydub import AudioSegment
from metrics import CorrelationMetric
import argparse

#first empty line is used to indicate lack of watermark (no modifcation)
supported_watermarks = ['','audioseal','wavmark','silentcipher']


def main():
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

def parse_technique_list(str,supported_list):
    if str == 'all':
        return supported_list
    used_techniques = []
    names = str.split(',')
    for name in names:
        if '-' in name: # treat it as range of values
            values = name.split('-')
            assert len(values) == 2, "Incorrectly defined range in options"
            assert values[0].isdigit(), "Wrong number for range of values"
            assert values[1].isdigit(), "Wrong number for range of values"
            used_techniques.extend(supported_list[values[0]:(values[1]+1)])
        elif name.isdigit():
            used_techniques.extend([name])
        else:
            # treat it as name of technique
            if name.lower() in supported_list:
                used_techniques.extend([name.lower()])
    used_techniques = list(dict.fromkeys(used_techniques))
    return used_techniques




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Tool for running audio watermarking and voice cloning pipeline")
    parser.add_argument("--samples",nargs='+',help="Input audio file or directory with audio files that will be used for watermarking")
    parser.add_argument("--watermarks",help="String comma-delimioted of numbers or names or simply \"all\" that indicate watermarking techniques to use in pipeline")
    parser.add_argument("--clones",help="String comma-delimioted of numbers or names or simply \"all\" that indicate voice cloning techniques to use in pipeline")
    parser.add_argument("--distortions",help="String comma-delimioted of numbers or names or simply \"all\" that indicate watermarking techniques to use in pipeline")
    main()
