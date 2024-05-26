from abc import ABC, abstractmethod

class BaseWatermark(ABC):

    @abstractmethod
    def add_watermark(self,audio):
        """
        For adding watermark to input audio.
        To be implemented by derived class.
        """
        pass
    
    #@abstractmethod
    #def preprocess(self,audio):
    #    """
    #    For preprocessing audio that later receives watermark.
    #    To be implemented by derived class.
    #    """
    #    pass