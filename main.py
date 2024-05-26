from watermarks import SimpleWatermark
import os
from pydub import AudioSegment
from metrics import CorrelationMetric

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

if __name__ == "__main__":
    main()
