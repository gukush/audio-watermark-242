from pydub import AudioSegment
from .base_watermark import BaseWatermark

class SimpleWatermark(BaseWatermark):
    def __init__(self, watermark_file, interval=5000):
        """
        Initializes the SimpleWatermark with a watermark audio file and interval.
        
        Args:
        watermark_file (str): Path to the watermark audio file.
        interval (int): Interval in milliseconds at which the watermark will be added.
        """
        self.watermark = AudioSegment.from_file(watermark_file)
        self.interval = interval

    def add_watermark(self, audio):
        """
        Adds a watermark to the given audio file at regular intervals.
        
        Args:
        audio_file (str): Path to the original audio file.

        Returns:
        AudioSegment: The watermarked audio segment.
        """
        
        duration = len(audio)
        
        # Overlay the watermark at regular intervals
        output = audio
        for i in range(0, duration, self.interval):
            if i + len(self.watermark) < duration:
                output = output.overlay(self.watermark, position=i)
        
        return output


    # unused for now
    def export_watermarked_audio(self, watermarked_audio, buffer, format='wav'):
        """
        Exports the watermarked audio to a specified file.

        Args:
        watermarked_audio (AudioSegment): The watermarked audio segment.
        export_path (str): Path to export the watermarked audio file.
        format (str): Format to export the audio (default: 'mp3').
        """
        watermarked_audio.export(buffer, format=format)
