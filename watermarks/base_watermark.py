from abc import ABC, abstractmethod

class BaseWatermark(ABC):
    name = "base"
    payload = None
    @abstractmethod
    def add_watermark(self,audio):
        """
        For adding watermark to input audio.
        To be implemented by derived class.
        """
        pass

    @abstractmethod
    def preprocess(self,input):
        """
        For preprocessing audio that later receives watermark.
        To be implemented by derived class.
        """
        pass

    @abstractmethod
    def detect_watermark(self,audio):
        """
        For detecting the watermark.
        To be implemented by derived class.
        """
        pass