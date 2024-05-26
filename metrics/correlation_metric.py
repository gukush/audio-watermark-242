from .base_metric import BaseMetric
from scipy.signal import correlate
from scipy.io import wavfile
import io
import numpy as np
# max_corr = np.max(correlation) / (np.linalg.norm(data_orig) * np.linalg.norm(data_water))
    
class CorrelationMetric(BaseMetric):
    
    def evaluate_quality(self, audio, watermarked_audio):
        """
        Receives AudioSegment object, converts it to wavfile,
        then computes the correlation metric.
        """
        # Conversion
        audio_buffer = io.BytesIO()
        watermarked_audio_buffer = io.BytesIO()
        audio.export(audio_buffer,format="wav")
        watermarked_audio.export(watermarked_audio_buffer,format="wav")
        rate_audio, data_audio = wavfile.read(audio_buffer)
        rate_watermarked, data_watermarked = wavfile.read(watermarked_audio_buffer)

        # Actuacl correlation calculation
        correlation = correlate(data_audio, data_watermarked)
        max_corr = np.max(correlation) / (np.linalg.norm(data_audio) * np.linalg.norm(data_watermarked))
        return max_corr